import os
import time
import torch
import pickle
from typing import Callable
from abc import ABC, abstractmethod
from experiments.currot.nadaraya_watson import GPUNadarayaWatson
from experiments.currot.gpu_currot_utils import SimpleWassersteinSuccessBuffer, ParameterSchedule
from experiments.currot.wasserstein_interpolation import GPUSamplingWassersteinInterpolation

Callback_Type = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], None]


class Parameter:

    def __init__(self, init_val: float, min_val: float, max_val: float) -> None:
        self.cur_val = init_val
        self.min_val = min_val
        self.max_val = max_val

    def get_value(self):
        return self.cur_val

    def set_value(self, new_val: float):
        self.cur_val = torch.clamp(new_val, self.min_val, self.max_val)


class ContextTransform(ABC):

    @abstractmethod
    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def color(self, x: torch.Tensor) -> torch.Tensor:
        pass


class IdentityTransform(ContextTransform):

    def whiten(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()

    def color(self, x: torch.Tensor) -> torch.Tensor:
        return x.clone()


class GPUCurrOT:

    def __init__(self, init_samples: torch.Tensor, target_sampler: Callable[[int], torch.Tensor],
                 metric_transform: Callable[[torch.Tensor], torch.Tensor], epsilon: ParameterSchedule,
                 constraint_fn: Callable[[torch.Tensor], torch.Tensor] = None, callback: Callback_Type = None,
                 success_buffer_size: int = None, buffer_size: int = None, context_transform: ContextTransform = None,
                 theta: float = 0.25 * torch.pi, old_optimizer: bool = False):
        """
        Creates a CurrOT sampler

        Parameters:
        init_samples: The intial samples from which the curriculum starts. In COLORED space (see context_transform for
                      details)
        target_sampler: A function that creates N samples from the target distribution in COLORED space, where N is an argument
        metric_transform: Transform the metrics that are tracked by the curriculum into one value. CurrOT aims to
                          keep this value above 0.
        epsilon: The step-size that should be allowed for the agent (wrapped by a class that allows scheduling based on the achieved
                 distance). This step size is defined in whitened transform!
        constraint_fn: An optional function that representes arbitrary constraints on the context vector (i.e.
                       representing a sub-region from the Euclidean embedding of the contexts). Those constraints are
                       expressed in WHITENED space.
        callback: An optional callback that is called by the interpolation module. Will obtain samples in COLORED space
        theta: The angle that defines the spherical cap of search directions for the optimization
        context_transform: A transformation that maps the context obtained from the samplers and returned to the learner
                           to an internal representation that is used for the sampling and performance prediction. This can
                           be used to realize the curriculum generation w.r.t. different metric spaces. For the case of
                           Mahalonbis distance, this module just represents a whitening/coloring tansform, hence the name of
                           WHITENED and COLORED space. CurrOT will internally work in WHITENED space. COLORED space is the
                           "observed" space by the learner and the callbacks.
        """
        # Ensure that the context transform is correctly applied
        if context_transform is not None:
            whitened_init_samples = context_transform.whiten(init_samples)
            whitened_target_sampler = lambda n: context_transform.whiten(target_sampler(n))
            if callback is None:
                whitened_callback = None
            else:
                whitened_callback = lambda *args: callback(context_transform.color(arg) for arg in args)
            self.context_transform = context_transform
        else:
            whitened_init_samples = init_samples
            whitened_target_sampler = target_sampler
            whitened_callback = callback
            self.context_transform = IdentityTransform()

        self.teacher = GPUSamplingWassersteinInterpolation(whitened_init_samples, whitened_target_sampler,
                                                           metric_transform,
                                                           epsilon, constraint_fn=constraint_fn,
                                                           callback=whitened_callback,
                                                           theta=theta, old_optimizer=old_optimizer)

        self.success_buffer = SimpleWassersteinSuccessBuffer(
            init_samples.shape[0] if success_buffer_size is None else success_buffer_size,
            metric_transform, squared_dist_fn="euclidean")

        self.buffer_size = init_samples.shape[0] if buffer_size is None else buffer_size
        self.context_buffer = None
        self.return_buffer = None
        self.last_model = None
        self.device = init_samples.device

    def update_distribution(self, contexts: torch.Tensor, metrics: torch.Tensor):
        # First thing we do is to transform our parameter to the WHITENED space
        contexts = self.context_transform.whiten(contexts)

        # This will only happen once after startup
        if self.context_buffer is None:
            self.context_buffer = torch.zeros((0, contexts.shape[1]), device=contexts.device)
            self.return_buffer = torch.zeros((0, metrics.shape[1]), device=metrics.device)

        # Evaluate the prediction quality
        if self.last_model is not None:
            predicted_metrics = self.teacher.metric_transform(self.last_model.predict_individual(contexts.contiguous()))
            predicted_success = predicted_metrics >= 0
            real_metrics = self.teacher.metric_transform(metrics)
            real_success = real_metrics >= 0
            # This means that if we predicted a success, we also want to witness a success
            accuracy = torch.sum(torch.logical_or(~predicted_success, real_success)) / predicted_success.shape[0]
            info = {"success_prediction_accuracy": accuracy,
                    "metric_precision": torch.mean(torch.abs(real_metrics - predicted_metrics))}
        else:
            info = {}

        t_up1 = time.time()
        target_samples = self.teacher.target_sampler(self.success_buffer.max_size)
        self.success_buffer.update(contexts, metrics, target_samples)

        self.context_buffer = torch.cat((self.context_buffer, contexts), dim=0)[-self.buffer_size:]
        self.return_buffer = torch.cat((self.return_buffer, metrics), dim=0)[-self.buffer_size:]

        model = GPUNadarayaWatson(self.context_buffer.clone(), self.return_buffer.clone(),
                                  0.3 * self.teacher.epsilon.get_value())
        t_up2 = time.time()

        t_mo1 = time.time()
        avg_perf = torch.mean(self.teacher.metric_transform(model.predict_individual(self.teacher.current_samples)),
                              dim=0)
        info["average_metric_value"] = avg_perf
        print(f"Current estimated performance: {avg_perf :.3e}")

        if len(self.success_buffer) >= 0.2 * self.teacher.current_samples.shape[0]:
            # Although here again we may have more success samples than initial ones, this
            # will be resolved in the ensure successful initial step of the interpolation
            n_samples = self.teacher.current_samples.shape[0]
            update_info = self.teacher.update_distribution(model, self.success_buffer.get_contexts(n_samples))
            info.update(update_info)
        else:
            info.update(self.teacher.empty_info())
            print("Not updating sampling distribution, as not enough successful samples are available")

        t_mo2 = time.time()

        # Store the model for the evaluation of the prediction quality
        self.last_model = model

        print("Total update took: %.3e (Buffer/Update: %.3e/%.3e)" % (t_mo2 - t_up1, t_up2 - t_up1, t_mo2 - t_mo1))
        return info

    def sample(self, env_ids: torch.Tensor):
        n = len(env_ids)
        n_samples = self.teacher.current_samples.shape[0]
        indices = torch.randperm(n_samples, device=self.teacher.current_samples.device)[:min(n, n_samples)]
        indices_extra = torch.randint(0, n_samples, size=(n - indices.shape[0],),
                                      device=self.teacher.current_samples.device)
        indices = torch.cat((indices, indices_extra), dim=0)
        return self.context_transform.color(self.teacher.current_samples[indices, :])

    def save(self, path):
        self.teacher.save(path)
        self.success_buffer.save(path)
        with open(os.path.join(path, "teacher_fail_buffer.pkl"), "wb") as f:
            pickle.dump((self.context_buffer.cpu().numpy(), self.return_buffer.cpu().numpy()), f)

    def load(self, path):
        self.teacher.load(path)
        self.success_buffer.load(path, self.device)
        if os.path.exists(os.path.join(path, "teacher_fail_buffer.pkl")):
            with open(os.path.join(path, "teacher_fail_buffer.pkl"), "rb") as f:
                fcb, frb = pickle.load(f)
            self.context_buffer = torch.from_numpy(fcb).to(self.device)
            self.return_buffer = torch.from_numpy(frb).to(self.device)
