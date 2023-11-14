import os
import torch
import pickle
import numpy as np
from typing import Callable
from scipy.optimize import linear_sum_assignment
from experiments.currot.auction import euclidean_low_mem_assignment
from experiments.currot.sampling import SphericalCapSampler, SphereSampler
from experiments.currot.gpu_currot_utils import ParameterSchedule


class OldSamplingDistanceOptimizer:

    def __init__(self, dim: int, epsilon: Callable[[], float],
                 constraint_fn: Callable[[torch.Tensor], torch.Tensor] = None, device="cpu") -> None:
        self.sphere_sampler = SphereSampler(dim, device=device)
        # This is a sampler on the half sphere
        self.cap_sampler = SphericalCapSampler(dim, 0.5 * torch.pi, cfd_disc=500, device=device)

        self.epsilon = epsilon
        if constraint_fn is None:
            self.constraint_fn = lambda x: torch.ones(x.shape[:-1], dtype=bool, device=x.device)
        else:
            self.constraint_fn = constraint_fn

    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], source_samples: torch.Tensor,
                 target_samples: torch.Tensor,
                 n_samples=250, epsilon_scales: torch.Tensor = None) -> torch.Tensor:
        source_performances = model(source_samples)
        perf_sufficient = source_performances >= 0
        source_distances = torch.norm(target_samples - source_samples, dim=-1)
        range_helper = torch.arange(source_samples.shape[0], device=source_samples.device)

        trust_region = self.epsilon() * torch.ones(source_samples.shape[0], device=source_samples.device)
        if epsilon_scales is not None:
            trust_region *= epsilon_scales
        trust_region = torch.minimum(trust_region, source_distances)

        descent_dir = target_samples - source_samples
        # Avoid division by zero if we are perfectly on the target
        descent_dir /= torch.clamp_min(source_distances[:, None], 1e-8)

        sphere_samples = self.sphere_sampler.sample_directions(n_samples)
        cap_samples = self.cap_sampler.sample_directions(n_samples)
        cap_samples = self.cap_sampler.rotate_directions(descent_dir, cap_samples)

        # Now we compute the individual search samples
        step_sizes = trust_region[:, None] * torch.pow(
            torch.rand(size=cap_samples.shape[:-1], device=source_samples.device), 1 / self.sphere_sampler.dim)
        use_cap = torch.logical_and(perf_sufficient, source_distances != 0.)
        search_samples = source_samples[:, None, :] + step_sizes[..., None] * torch.where(use_cap[:, None, None],
                                                                                          cap_samples, sphere_samples)

        # Next we need to evaluate the performance and constraint function
        constraint_fulfilled = self.constraint_fn(search_samples)
        performances = model(search_samples)
        performance_reached = performances >= 0.
        distances = torch.norm(target_samples[:, None, :] - search_samples, dim=-1)

        feasible = torch.logical_and(constraint_fulfilled, performance_reached)
        any_feasible = torch.any(feasible, dim=-1)

        # We first get the best possible distance (masking out any of the infeasible samples)
        min_distance = torch.argmin(torch.where(feasible, distances, torch.inf), dim=-1)

        # We also get the maximum performing sample (masking out any of the samples that do not fulfill the constraint)
        max_perf = torch.argmax(torch.where(constraint_fulfilled, performances, -torch.inf), dim=-1)

        # The new sample is now either the feasible minimum distance one or the maximum performance one (note that we add the
        # initial sample which guarantees us that in the worst case we just get the old sample)
        best_idx = torch.where(any_feasible, min_distance, max_perf)

        # Finally, we update the information for the next rounds (making sure that we do not use samples that are not in the
        # constrained region)
        new_fulfilled = constraint_fulfilled[range_helper, best_idx]
        new_samples = torch.where(new_fulfilled[:, None], search_samples[range_helper, best_idx], source_samples)
        new_performances = torch.where(new_fulfilled, performances[range_helper, best_idx], source_performances)
        new_distances = torch.where(new_fulfilled, distances[range_helper, best_idx], source_distances)

        # We finally compute a success rate, i.e. the rate with which we either improved performance (finding a feasible point)
        # from a previously infeasible one or decreased target distance
        dist_decreased = new_distances < source_distances

        # If we did not find a better sample, we just keep the source sample
        new_samples = torch.where(torch.logical_and(perf_sufficient, ~dist_decreased)[:, None], source_samples,
                                  new_samples)

        optimization_success = torch.where(perf_sufficient, dist_decreased, new_performances > source_performances)

        # This is the rate of particles that either reached their goal or basically saturated the trust-region (can be used to decide upon
        # increasing the trust-region)
        saturation_rate = torch.mean(torch.logical_or(new_distances <= 1e-3,
                                                      torch.norm(source_samples - new_samples,
                                                                 dim=-1) >= 0.95 * trust_region).float())

        return new_samples, new_distances, torch.mean(optimization_success.float()), saturation_rate


class SamplingDistanceOptimizer:

    def __init__(self, dim: int, epsilon: Callable[[], float],
                 constraint_fn: Callable[[torch.Tensor], torch.Tensor] = None, device="cpu",
                 theta: float = 0.25 * torch.pi) -> None:
        self.sphere_sampler = SphereSampler(dim, device=device)
        self.cap_sampler = SphericalCapSampler(dim, theta, cfd_disc=500, device=device)

        self.epsilon = epsilon
        if constraint_fn is None:
            self.constraint_fn = lambda x: torch.ones(x.shape[:-1], dtype=bool, device=x.device)
        else:
            self.constraint_fn = constraint_fn

    def _get_trust_regions(self, init_distances, epsilon_scales=None):
        init_distances_squared = init_distances ** 2
        init_trust_region_squared = self.epsilon() * torch.ones(init_distances.shape[0], device=init_distances.device)
        if epsilon_scales is not None:
            init_trust_region_squared *= epsilon_scales
        init_trust_region_squared = init_trust_region_squared ** 2

        # We do an initial balancing of the trust-region (to give the required extra step size to the samples that need
        # it)
        diffs_squared = init_trust_region_squared - init_distances_squared
        slack_mask = diffs_squared > 0
        total_slack = torch.sum(diffs_squared[slack_mask])
        if total_slack > 0:
            missing_mask = diffs_squared < 0
            missing = torch.abs(diffs_squared[missing_mask])
            total_missing = torch.sum(missing)
            if total_missing <= total_slack:
                trust_region_squared = init_distances_squared.clone()
                remaining_slack = torch.sum(init_trust_region_squared) - torch.sum(trust_region_squared)
                assert remaining_slack >= 0
                trust_region_squared += (trust_region_squared / torch.sum(trust_region_squared)) * remaining_slack
            else:
                trust_region_squared = init_trust_region_squared.clone()
                trust_region_squared[slack_mask] = init_distances_squared[slack_mask]
                percentage = missing / total_missing
                trust_region_squared[missing_mask] += total_slack * percentage

            assert torch.sqrt(torch.mean(trust_region_squared)) <= self.epsilon() + 1e-5
            return torch.sqrt(trust_region_squared)
        else:
            return torch.sqrt(init_trust_region_squared)

    def __call__(self, model: Callable[[torch.Tensor], torch.Tensor], source_samples: torch.Tensor,
                 target_samples: torch.Tensor, n_iter=5, n_samples=50,
                 epsilon_scales: torch.Tensor = None) -> torch.Tensor:
        init_performances = model(source_samples)
        init_perf_sufficient = init_performances >= 0
        init_distances = torch.norm(target_samples - source_samples, dim=-1)
        range_helper = torch.arange(source_samples.shape[0], device=source_samples.device)

        current_samples = source_samples
        distances = init_distances
        performances = init_performances
        perf_sufficient = init_perf_sufficient
        trust_region = self._get_trust_regions(init_distances, epsilon_scales=epsilon_scales)

        for i in range(n_iter):
            descent_dir = target_samples - current_samples
            # Avoid division by zero if we are perfectly on the target
            descent_dir /= torch.clamp_min(distances[:, None], 1e-8)

            sphere_samples = self.sphere_sampler.sample_directions(n_samples)
            cap_samples = self.cap_sampler.sample_directions(n_samples)
            cap_samples = self.cap_sampler.rotate_directions(descent_dir, cap_samples)

            # We compute the maximum step_size
            current_step_sizes = trust_region - torch.norm(source_samples - current_samples, dim=-1)
            if torch.any(current_step_sizes < (-0.005 * trust_region)[:, None]):
                print("Warning!")
            current_step_sizes = torch.clamp_min(current_step_sizes, 0.)
            current_step_sizes = torch.minimum(distances, current_step_sizes)

            # Now we compute the individual search samples
            step_sizes = torch.rand(size=cap_samples.shape[:-1], device=source_samples.device) * current_step_sizes[:,
                                                                                                 None]
            # If we are right on point, and have a true 0 vector, the rotate_directions method of the cap sampler produces nans
            use_cap = torch.logical_and(perf_sufficient, distances != 0.)
            search_samples = current_samples[:, None, :] + step_sizes[..., None] * torch.where(use_cap[:, None, None],
                                                                                               cap_samples,
                                                                                               sphere_samples)

            # We always provide the current sample as well as the full step to the set of search points.
            # Given that we assume that the given initial point fulfills the constraint, we can be sure that our final point will for sure as well
            search_samples = torch.cat((search_samples,
                                        current_samples[:, None, :]), dim=1)

            # Next we need to evaluate the performance and constraint function
            search_constraint_fulfilled = self.constraint_fn(search_samples)
            search_performances = model(search_samples)
            search_distances = torch.norm(target_samples[:, None, :] - search_samples, dim=-1)

            feasible = torch.logical_and(search_constraint_fulfilled, search_performances >= 0.)
            any_feasible = torch.any(feasible, dim=-1)

            # We first get the best possible distance (masking out any of the infeasible samples)
            min_distance = torch.argmin(torch.where(feasible, search_distances, torch.inf), dim=-1)

            # We also get the maximum performing sample (masking out any of the samples that do not fulfill the constraint)
            max_perf = torch.argmax(torch.where(search_constraint_fulfilled, search_performances, -torch.inf), dim=-1)

            # The new sample is now either the feasible minimum distance one or the maximum performance one (note that we add the
            # initial sample which guarantees us that in the worst case we just get the old sample)
            best_idx = torch.where(any_feasible, min_distance, max_perf)

            # Finally, we update the information for the next rounds (only using samples that fulfill the constraint)
            constraint_fulfilled = search_constraint_fulfilled[range_helper, best_idx]
            current_samples = torch.where(constraint_fulfilled[:, None], search_samples[range_helper, best_idx],
                                          current_samples)
            performances = torch.where(constraint_fulfilled, search_performances[range_helper, best_idx], performances)
            perf_sufficient = torch.where(constraint_fulfilled, performances >= 0., perf_sufficient)
            distances = torch.where(constraint_fulfilled, search_distances[range_helper, best_idx], distances)

        # We finally compute a success rate, i.e. the rate with which we either improved performance (finding a feasible point)
        # from a previously infeasible one or decreased target distance
        dist_decreased = distances < init_distances
        optimization_success = torch.where(init_perf_sufficient, dist_decreased, performances > init_performances)
        if torch.any(torch.norm(source_samples - current_samples, dim=-1) > 1.005 * trust_region):
            print("Warning!")

        # This is the rate of particles that either reached their goal or basically saturated the trust-region (can be used to decide upon
        # increasing the trust-region)
        saturation_rate = torch.mean(torch.logical_or(distances <= 1e-3,
                                                      torch.norm(source_samples - current_samples,
                                                                 dim=-1) >= 0.95 * trust_region).float())

        return current_samples, distances, torch.mean(optimization_success.float()), saturation_rate


class GPUSamplingWassersteinInterpolation:

    def __init__(self, init_samples: torch.Tensor, target_sampler: Callable[[int], torch.Tensor],
                 metric_transform: Callable[[torch.Tensor], torch.Tensor], epsilon: ParameterSchedule,
                 callback: Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None] = None,
                 constraint_fn: Callable[[torch.Tensor], torch.Tensor] = None,
                 theta: float = 0.25 * torch.pi, old_optimizer: bool = False) -> None:
        self.current_samples = init_samples.contiguous()
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler
        self.metric_transform = metric_transform
        self.epsilon = epsilon
        self.callback = callback
        if old_optimizer:
            self.optimizer = OldSamplingDistanceOptimizer(self.dim, self.epsilon.get_value, constraint_fn,
                                                          init_samples.device)
        else:
            self.optimizer = SamplingDistanceOptimizer(self.dim, self.epsilon.get_value, constraint_fn,
                                                       init_samples.device, theta=theta)

    def ensure_successful_initial(self, model, init_samples, success_samples):
        success_assignment = euclidean_low_mem_assignment(init_samples, success_samples, epsilon=0.02)

        performance_reached = self.metric_transform(model.predict_individual(init_samples)) >= 0.
        assigned_samples = success_samples[success_assignment[1][~performance_reached]]
        backup = ~performance_reached
        init_samples[backup, :] = assigned_samples
        performance_reached[~performance_reached] = self.metric_transform(
            model.predict_individual(assigned_samples)) >= 0.

        return init_samples, performance_reached, backup

    def empty_info(self):
        target_samples = self.target_sampler(self.n_samples)
        assignments = euclidean_low_mem_assignment(self.current_samples, target_samples, epsilon=0.02)
        dists = torch.norm(self.current_samples[assignments[0]] - target_samples[assignments[1]], dim=-1)

        info = {"optimization_success": torch.zeros(0, device=self.current_samples.device),
                "epsilon": torch.zeros(0, device=self.current_samples.device),
                "wasserstein_distance": torch.sqrt(torch.mean(torch.square(dists))),
                "saturation_rate": torch.zeros(0, device=self.current_samples.device)}
        return info

    def update_distribution(self, model, success_samples, debug=False):
        init_samples, performance_reached, backup = self.ensure_successful_initial(model, self.current_samples.clone(),
                                                                                   success_samples)
        target_samples = self.target_sampler(self.n_samples)
        assignments = euclidean_low_mem_assignment(init_samples, target_samples, epsilon=0.02)

        init_samples = init_samples[assignments[0]]
        target_samples = target_samples[assignments[1]]

        new_samples, new_distances, optimization_success, saturation_rate = self.optimizer(
            lambda x: self.metric_transform(model.predict_individual(x)),
            init_samples, target_samples)
        new_wdist = torch.sqrt(torch.mean(torch.square(new_distances)))
        print(f"New Wasserstein Distance: {new_wdist}")
        print(f"Optimization Success Rate: {optimization_success}")
        print(f"Saturation Rate: {saturation_rate}")

        # Update the epsilon parameter with the new wasserstein distance
        if saturation_rate > 0.8:
            self.epsilon.step()
        # self.epsilon.update(new_wdist)

        info = {"optimization_success": optimization_success,
                "epsilon": self.epsilon.get_value(),
                "wasserstein_distance": new_wdist,
                "saturation_rate": saturation_rate}

        if self.callback is not None:
            self.callback(self.current_samples, new_samples, success_samples, target_samples)

        self.current_samples = new_samples.contiguous()

        return info

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples.detach().cpu().numpy(), self.epsilon), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = torch.from_numpy(tmp[0]).to(self.current_samples.device)
            self.n_samples = self.current_samples.shape[0]

            self.epsilon = tmp[1]
            self.optimizer.epsilon = self.epsilon.get_value


class SamplingWassersteinInterpolation:

    def __init__(self, init_samples, target_sampler, perf_lb, epsilon, bounds, callback=None):
        self.current_samples = init_samples
        self.n_samples, self.dim = self.current_samples.shape
        self.target_sampler = target_sampler
        self.bounds = bounds
        self.perf_lb = perf_lb
        self.epsilon = epsilon
        self.callback = callback

    def sample_ball(self, targets, samples=None, half_ball=None, n=100):
        if samples is None:
            samples = self.current_samples

        # Taken from http://extremelearning.com.au/how-to-generate-uniformly-random-points-on-n-spheres-and-n-balls/
        # Method 20
        direction = np.random.normal(0, 1, (n, self.dim))
        norm = np.linalg.norm(direction, axis=-1, keepdims=True)
        r = np.power(np.random.uniform(size=(n, 1)), 1. / self.dim)

        # We only consider samples that decrease the distance objective (i.e. are aligned with the direction)
        noise = r * (direction / norm)
        dirs = targets - samples
        dir_norms = np.einsum("ij,ij->i", dirs, dirs)
        noise_projections = np.einsum("ij,kj->ik", dirs / dir_norms[:, None], noise)

        projected_noise = np.where((noise_projections > 0)[..., None], noise[None, ...],
                                   noise[None, ...] - 2 * noise_projections[..., None] * dirs[:, None, :])
        if half_ball is not None:
            projected_noise[~half_ball] = noise

        scales = np.minimum(self.epsilon, np.sqrt(dir_norms))[:, None, None]
        return np.clip(samples[..., None, :] + scales * projected_noise, self.bounds[0], self.bounds[1])

    @staticmethod
    def visualize_particles(init_samples, particles, performances):
        if particles.shape[-1] != 2:
            raise RuntimeError("Can only visualize 2D data")

        import matplotlib.pyplot as plt
        f = plt.figure()
        ax = f.gca()
        scat = ax.scatter(particles[0, :, 0], particles[0, :, 1], c=performances[0, :])
        ax.scatter(init_samples[0, 0], init_samples[0, 1], marker="x", c="red")
        plt.colorbar(scat)
        plt.show()

    def ensure_successful_initial(self, model, init_samples, success_samples):
        squared_dists = np.sum(np.square(init_samples[:, None, :] - success_samples[None, :, :]), axis=-1)
        success_assignment = linear_sum_assignment(squared_dists, maximize=False)

        performance_reached = model.predict_individual(init_samples) >= self.perf_lb
        assigned_samples = success_samples[success_assignment[1][~performance_reached]]
        init_samples[~performance_reached, :] = assigned_samples
        performance_reached[~performance_reached] = model.predict_individual(assigned_samples) >= self.perf_lb

        return init_samples, performance_reached

    def update_distribution(self, model, success_samples, debug=False):
        init_samples, performance_reached = self.ensure_successful_initial(model, self.current_samples.copy(),
                                                                           success_samples)
        target_samples = self.target_sampler(self.n_samples)
        if debug:
            target_samples_true = target_samples.copy()
        assignments = linear_sum_assignment(np.sum(np.square(init_samples[:, None] - target_samples[None, :]), axis=-1))
        init_samples = init_samples[assignments[0]]
        target_samples = target_samples[assignments[1]]
        # movements = sliced_wasserstein(init_samples, target_samples, grad=True)[1]
        # target_samples = init_samples + movements
        particles = self.sample_ball(target_samples, samples=init_samples, half_ball=performance_reached)

        distances = np.linalg.norm(particles - target_samples[:, None, :], axis=-1)
        performances = model.predict_individual(particles)
        if debug:
            self.visualize_particles(init_samples, particles, performances)

        mask = performances > self.perf_lb
        solution_possible = np.any(mask, axis=-1)
        distances[~mask] = np.inf
        opt_idxs = np.where(solution_possible, np.argmin(distances, axis=-1), np.argmax(performances, axis=-1))
        new_samples = particles[np.arange(0, self.n_samples), opt_idxs]

        print(f"New Wasserstein Distance: {np.sqrt(np.mean(np.sum(np.square(new_samples - target_samples), axis=-1)))}")

        if debug:
            vis_idxs = np.random.randint(0, target_samples.shape[0], size=50)
            import matplotlib.pyplot as plt
            xs, ys = np.meshgrid(np.linspace(0, 9, num=150), np.linspace(0, 6, num=100))
            zs = model.predict_individual(np.stack((xs, ys), axis=-1))
            ims = plt.imshow(zs, extent=[0, 9, 0, 6], origin="lower")
            plt.contour(xs, ys, zs, [180])
            plt.colorbar(ims)

            plt.scatter(target_samples_true[vis_idxs, 0], target_samples_true[vis_idxs, 1], marker="x", color="red")
            plt.scatter(self.current_samples[vis_idxs, 0], self.current_samples[vis_idxs, 1], marker="o", color="C0")
            plt.scatter(init_samples[vis_idxs, 0], init_samples[vis_idxs, 1], marker="o", color="C2")
            plt.scatter(new_samples[vis_idxs, 0], new_samples[vis_idxs, 1], marker="o", color="C1")
            plt.xlim([0, 9])
            plt.ylim([0, 6])
            plt.show()

        if self.callback is not None:
            self.callback(self.current_samples, new_samples, success_samples, target_samples)

        self.current_samples = new_samples

    def save(self, path):
        with open(os.path.join(path, "teacher.pkl"), "wb") as f:
            pickle.dump((self.current_samples, self.perf_lb, self.epsilon), f)

    def load(self, path):
        with open(os.path.join(path, "teacher.pkl"), "rb") as f:
            tmp = pickle.load(f)

            self.current_samples = tmp[0]
            self.n_samples = self.current_samples.shape[0]

            self.perf_lb = tmp[1]
            self.epsilon = tmp[2]
