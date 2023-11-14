import os
import torch
import pickle
from abc import ABC, abstractmethod
from typing import Tuple, Any, List, NoReturn, Callable, Union
from experiments.currot.auction import assignment, euclidean_low_mem_assignment


class ParameterSchedule:

    def __init__(self, epsilons: torch.Tensor) -> None:
        self.epsilons = epsilons

        if len(self.epsilons.shape) == 0:
            self.epsilons = self.epsilons[None]

        # We assume that the list is sorted from big to small
        self.cur_index = 0

    def get_value(self):
        return self.epsilons[self.cur_index]

    def step(self):
        if self.cur_index < self.epsilons.shape[0] - 1:
            self.cur_index += 1


class GPUTruncatedGaussianSampling:

    def __init__(self, search_stds: torch.Tensor, min_stds: torch.Tensor,
                 constraint_fn: Callable[[torch.Tensor], torch.Tensor] = None, device="cpu"):
        self.min_ret = None
        self.delta_stds = search_stds
        self.min_stds = min_stds
        self.constraint_fn = constraint_fn
        self.device = device

    def get_data(self):
        if self.min_ret is None:
            return None
        else:
            return self.min_ret.cpu().numpy()

    def set_data(self, data):
        if data is None:
            self.min_ret = data
        else:
            self.min_ret = torch.from_numpy(data).to(self.device)

    def __call__(self, contexts: torch.Tensor, values: torch.Tensor, n_samples=100):
        if self.min_ret is None:
            # The + 1e-8 is for numerical reasons (if the minimum happens to be hexactly zero - yep this happened)
            self.min_ret = (torch.min(values) - 1e-8)
        else:
            self.min_ret = torch.minimum(self.min_ret, torch.min(values) - 1e-8)
        # We want to keep the transformed metric value above 0.
        var_scales = torch.clip(values, -torch.inf, 0) / self.min_ret
        stds = self.min_stds[None, :] + var_scales[:, None] * self.delta_stds[None, :]

        contexts = contexts[:, None, :] + stds[:, None, :] * torch.randn(
            size=(contexts.shape[0], n_samples, contexts.shape[1]),
            device=contexts.device)
        valid = self.constraint_fn(contexts)
        sample_idx = torch.argmax(valid.float(), dim=1)

        helper = torch.arange(contexts.shape[0], device=contexts.device)
        if not torch.all(valid[helper, sample_idx]):
            raise RuntimeError("Could not sample a valid next point for each sample")

        return contexts[helper, sample_idx]


class GPUAbstractSuccessBuffer(ABC):

    def __init__(self, n: int, metric_transform: Callable[[torch.Tensor], torch.Tensor],
                 sampler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]):
        self.max_size = n
        self.metric_transform = metric_transform
        self.contexts = None
        self.metrics = None
        self.values = None
        self.sampler = sampler
        self.delta_reached = False

    @abstractmethod
    def update_delta_not_reached(self, new_contexts: torch.Tensor, new_metrics: torch.Tensor, new_values: torch.Tensor,
                                 current_samples: torch.Tensor) -> Tuple[bool, torch.Tensor, torch.Tensor, List[bool]]:
        pass

    @abstractmethod
    def update_delta_reached(self, new_contexts: torch.Tensor, new_metrics: torch.Tensor, new_values: torch.Tensor,
                             current_samples: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, List[bool]]:
        pass

    def __len__(self):
        if self.contexts is None:
            return 0
        else:
            return self.contexts.shape[0]

    def update(self, contexts: torch.Tensor, metrics: torch.Tensor, target_samples: torch.Tensor):
        assert contexts.shape[0] <= self.max_size

        if self.contexts is None:
            self.contexts = torch.zeros((0, contexts.shape[1]), device=contexts.device)

        if self.metrics is None:
            self.metrics = torch.zeros((0, metrics.shape[1]), device=contexts.device)

        if self.values is None:
            self.values = torch.zeros((0,), device=contexts.device)

        values = self.metric_transform(metrics)
        if not self.delta_reached:
            self.delta_reached, self.contexts, self.metrics, self.values, mask = \
                self.update_delta_not_reached(contexts, metrics, values, target_samples)
        else:
            self.contexts, self.metrics, self.values, mask = \
                self.update_delta_reached(contexts, metrics, values, target_samples)

        info = {"success_percentage": torch.sum(values >= 0) / values.shape[0],
                "success_buffer_inclusion": 1 - (torch.sum(mask) / mask.shape[0])}
        return contexts[mask, :], metrics[mask], info

    def read_train(self):
        return self.contexts.clone(), self.metrics.clone()

    def read_update(self):
        # If we did not yet reach the desired threshold we enforce exploration by scaling the exploration noise w.r.t.
        # the distance to the desired threshold value
        if not self.delta_reached:
            offset = self.metrics.shape[0] // 2
            sub_values = self.values[offset:]
            sub_contexts = self.contexts[offset:, :]

            # Do a resampling based on the achieved rewards (favouring higher rewards to be resampled)
            probs = sub_values - self.values[offset - 1]
            norm = torch.sum(probs)
            if norm == 0:
                probs = torch.ones(sub_values.shape[0], device=sub_values.device) / sub_values.shape[0]
            else:
                probs = probs / norm

            sample_idxs = torch.argmax((torch.rand(self.max_size, device=self.values.device)[:, None] < torch.cumsum(
                probs, dim=0)[None, :]).float(), dim=-1)
            sampled_contexts = sub_contexts[sample_idxs, :]
            sampled_values = sub_values[sample_idxs]
        else:
            to_fill = self.max_size - self.metrics.shape[0]
            add_idxs = torch.randint(0, self.metrics.shape[0], size=(to_fill,), device=self.metrics.device)
            sampled_contexts = torch.cat((self.contexts, self.contexts[add_idxs, :]), dim=0)
            sampled_values = torch.cat((self.values, self.values[add_idxs]), dim=0)

        return self.sampler(sampled_contexts, sampled_values)

    def get_data(self) -> Any:
        return None

    def set_data(self, data: Any) -> NoReturn:
        pass

    def save(self, path):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "wb") as f:
            pickle.dump((self.max_size, self.contexts.cpu().numpy(), self.values.cpu().numpy(),
                         self.metrics.cpu().numpy(), self.delta_reached, self.get_data(), self.sampler.get_data()), f)

    def load(self, path, device):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "rb") as f:
            self.max_size, contexts, values, metrics, self.delta_reached, subclass_data, sampler_data = pickle.load(f)
        self.contexts = torch.from_numpy(contexts).to(device)
        self.values = torch.from_numpy(values).to(device)
        self.metrics = torch.from_numpy(metrics).to(device)
        self.sampler.set_data(sampler_data)
        self.set_data(subclass_data)


class SimpleWassersteinSuccessBuffer:

    def __init__(self, n: int, metric_transform: Callable[[torch.Tensor], torch.Tensor],
                 squared_dist_fn: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str] = "euclidean"):
        self.max_size = n
        self.metric_transform = metric_transform
        self.contexts = None
        self.metrics = None
        self.squared_dist_fn = squared_dist_fn

    def __len__(self):
        if self.contexts is None:
            return 0
        else:
            return self.contexts.shape[0]

    def update(self, contexts: torch.Tensor, metrics: torch.Tensor, target_samples: torch.Tensor):
        if self.contexts is None:
            self.contexts = torch.zeros((0, contexts.shape[1]), device=contexts.device)
            self.metrics = torch.zeros((0, metrics.shape[1]), device=metrics.device)

        # Compute the new successful samples
        values = self.metric_transform(metrics)
        mask = values >= 0.
        n_new = torch.sum(mask)

        print(f"Updating success buffer with {n_new} samples.")
        if n_new > 0:
            extended_contexts = torch.cat((self.contexts, contexts[mask, :]), dim=0)
            extended_metrics = torch.cat((self.metrics, metrics[mask]), dim=0)
            if extended_contexts.shape[0] >= self.max_size:
                # At this stage we use the optimizer
                if self.squared_dist_fn == "euclidean":
                    # For this implementation, epsilon is relative to the spread of the squared distances (since we
                    # want to avoid computaion all distances explicitly in the first place)
                    assignments = euclidean_low_mem_assignment(extended_contexts, target_samples, epsilon=0.02)
                else:
                    squared_dists = self.squared_dist_fn(extended_contexts[:, None, :], target_samples[None, :, :])
                    assignments = assignment(squared_dists,
                                             epsilon=0.02 * (torch.max(squared_dists) - torch.min(squared_dists)))

                ret_idxs = assignments[0]
                self.contexts = extended_contexts[ret_idxs, :]
                self.metrics = extended_metrics[ret_idxs]
            else:
                self.contexts = extended_contexts
                self.metrics = extended_metrics

    def get_contexts(self, n):
        if self.contexts.shape[0] >= n:
            return self.contexts.clone()
        else:
            n_double = n - self.contexts.shape[0]
            return torch.cat((self.contexts, self.contexts[torch.randint(0, self.contexts.shape[0], (n_double,)), :]))

    def save(self, path):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "wb") as f:
            pickle.dump((self.max_size, self.contexts.cpu().numpy(), self.metrics.cpu().numpy()), f)

    def load(self, path, device):
        with open(os.path.join(path, "teacher_success_buffer.pkl"), "rb") as f:
            self.max_size, contexts, metrics = pickle.load(f)
        self.contexts = torch.from_numpy(contexts).to(device)
        self.metrics = torch.from_numpy(metrics).to(device)


class GPUWassersteinSuccessBuffer(GPUAbstractSuccessBuffer):

    def __init__(self, n: int, metric_transform: Callable[[torch.Tensor], torch.Tensor],
                 sampler: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 squared_dist_fn: Union[Callable[[torch.Tensor, torch.Tensor], torch.Tensor], str] = "euclidean"):
        super().__init__(n, metric_transform, sampler)
        self.squared_dist_fn = squared_dist_fn

    def update_delta_not_reached(self, contexts: torch.Tensor, metrics: torch.Tensor, values: torch.Tensor,
                                 target_samples: torch.Tensor) -> Tuple[
        bool, torch.Tensor, torch.Tensor, torch.Tensor, List[bool]]:
        # Only add samples that have a higher return than the median return in the buffer (we do >= here to allow
        # for binary rewards to work)
        if self.metrics.shape[0] > 0:
            med_idx = self.metrics.shape[0] // 2
            mask = values >= self.values[med_idx]
        else:
            mask = torch.ones_like(values, dtype=torch.bool)
            med_idx = 0
        n_new = torch.sum(mask)
        print("Improving buffer quality with %d samples" % n_new)

        # We do not want to shrink the buffer
        offset_idx = med_idx + 1
        if n_new < offset_idx:
            offset_idx = n_new

        new_metrics = torch.cat((metrics[mask], self.metrics[offset_idx:]), dim=0)
        new_values = torch.cat((values[mask], self.values[offset_idx:]), dim=0)
        new_contexts = torch.cat((contexts[mask, :], self.contexts[offset_idx:, :]), dim=0)
        sort_idxs = torch.argsort(new_values)

        # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
        assert self.contexts.shape[0] <= new_contexts.shape[0]
        assert new_contexts.shape[0] == new_metrics.shape[0]

        # These are the indices of the tasks that have NOT been added to the buffer (so the negation of the mas)
        rem_mask = ~mask

        # Ensure that we are not larger than the maximum size
        if new_metrics.shape[0] > self.max_size:
            sort_idxs = sort_idxs[-self.max_size:]
            # Since we are clipping potentially removing some of the data chunks we need to update the remainder mask
            # Since we add the new samples at the beginning of the new buffers, we are interested whether the idxs
            # in [0, n_new) are still in the sort_idxs array. If this is NOT the case, then the sample has NOT been
            # added to the buffer.
            removed_samples = torch.topk(sort_idxs, k=new_metrics.shape[0] - self.max_size, largest=False).indices
            removed_samples = removed_samples[removed_samples < n_new]
            is_removed = torch.zeros(n_new, dtype=torch.bool, device=mask.device)
            is_removed[removed_samples] = True
            rem_mask[mask] = is_removed

        new_metrics = new_metrics[sort_idxs]
        new_values = new_values[sort_idxs]
        new_contexts = new_contexts[sort_idxs, :]

        new_delta_reached = new_values[new_values.shape[0] // 2] >= 0.
        return new_delta_reached, new_contexts, new_metrics, new_values, rem_mask

    def update_delta_reached(self, contexts: torch.Tensor, metrics: torch.Tensor, values: torch.Tensor,
                             current_samples: torch.Tensor) -> \
            Tuple[torch.Tensor, torch.Tensor, List[bool]]:
        # Compute the new successful samples
        mask = values >= 0.
        n_new = torch.sum(mask)

        if n_new > 0:
            remove_mask = self.values < 0.
            if not torch.any(remove_mask) and self.metrics.shape[0] >= self.max_size:
                extended_contexts = torch.cat((self.contexts, contexts[mask, :]), dim=0)
                extended_metrics = torch.cat((self.metrics, metrics[mask]), dim=0)
                extended_values = torch.cat((self.values, values[mask]), dim=0)

                # At this stage we use the optimizer
                if self.squared_dist_fn == "euclidean":
                    # For this implementation, epsilon is relative to the spread of the squared distances (since we
                    # want to avoid computaion all distances explicitly in the first place)
                    assignments = euclidean_low_mem_assignment(extended_contexts, current_samples, epsilon=0.02)
                else:
                    squared_dists = self.squared_dist_fn(extended_contexts[:, None, :], current_samples[None, :, :])
                    assignments = assignment(squared_dists,
                                             epsilon=0.02 * (torch.max(squared_dists) - torch.min(squared_dists)))

                ret_idxs = assignments[0]
                new_contexts = extended_contexts[ret_idxs, :]
                new_metrics = extended_metrics[ret_idxs]
                new_values = extended_values[ret_idxs]

                # We update the mask to indicate only the kept samples
                kept_idx = ret_idxs[ret_idxs >= self.contexts.shape[0]] - self.contexts.shape[0]
                are_kept = torch.zeros(n_new, dtype=torch.bool, device=ret_idxs.device)
                are_kept[kept_idx] = True
                mask[mask.clone()] = are_kept

                print(f"Updated success buffer with {n_new} samples.")
            else:
                # We replace the unsuccessful samples by the successful ones
                if n_new < torch.sum(remove_mask):
                    remove_idxs = torch.topk(self.values, k=n_new, largest=False, sorted=False).indices
                    remove_mask = torch.zeros(self.values.shape[0], dtype=bool, device=self.metrics.device)
                    remove_mask[remove_idxs] = True

                new_metrics = torch.cat((metrics[mask], self.metrics[~remove_mask]), dim=0)
                new_values = torch.cat((values[mask], self.values[~remove_mask]), dim=0)
                new_contexts = torch.cat((contexts[mask, :], self.contexts[~remove_mask, :]), dim=0)

                if new_metrics.shape[0] > self.max_size:
                    new_metrics = new_metrics[:self.max_size]
                    new_values = new_values[:self.max_size]
                    new_contexts = new_contexts[:self.max_size, :]

                # Ensure that the buffer is only growing, never shrinking and that all the buffer sizes are consistent
                assert self.contexts.shape[0] <= new_contexts.shape[0]
                assert new_contexts.shape[0] == new_metrics.shape[0]
                print(f"Added {n_new} success samples to the success buffer.")
        else:
            new_contexts = self.contexts
            new_metrics = self.metrics
            new_values = self.values

        return new_contexts, new_metrics, new_values, ~mask
