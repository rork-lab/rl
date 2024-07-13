# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import importlib.util

from typing import Any, Dict, Optional, Sequence, Union

from torch import Tensor

from torchrl.record.loggers.common import Logger

_has_clearml = importlib.util.find_spec("clearml") is not None
_has_omegaconf = importlib.util.find_spec("omegaconf") is not None


class ClearMLLogger(Logger):
    """Wrapper for the ClearML logger.

    Args:
        exp_name (str): The name of the experiment.
        task_name (str): The name of the task.
    """

    def __init__(
        self,
        project_name: str,
        task_name: str,
        tags: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:

        self._clearml_kwargs = {
            "project_name": project_name,
            "task_name": task_name,
            "tags": tags,
            **kwargs,
        }

        super().__init__(exp_name=task_name, log_dir=project_name)
        self.video_log_counter = 0

    def _create_experiment(self) -> "ClearMLLogger":  # noqa

        """Creates an ClearML experiment.

        Returns:
            clearml.Task.init: The clearml experiment object.
        """

        if not _has_clearml:
            raise ImportError("ClearML is not installed")

        import clearml

        # Create project if it doesn't exist
        # try:
        #     self.task = clearml.Task.get_task(
        #         project_name=self._clearml_kwargs["project_name"],
        #         task_name=self._clearml_kwargs["task_name"])
        #     print(self.task)
        # except ValueError:
        #     print("ERRROR")
        self.task = clearml.Task.init(
                project_name=self._clearml_kwargs["project_name"],
                task_name=self._clearml_kwargs["task_name"],
                tags=self._clearml_kwargs["tags"])
        print(self.task)
        return self.task

    def log_scalar(self, name: str, value: float, step: Optional[int] = None,
                   series: str = "",
                   is_single: bool=False) -> None:
        """Logs a scalar value to clearml.

        Args:
            name (str): The name of the scalar.
            value (float): The value of the scalar.
            step (int, optional): The step at which the scalar is logged.
                Defaults to None.
        """
        from clearml import Logger

        if is_single:
            Logger.current_logger().report_single_value(name=name, value=value)
        else:
            Logger.current_logger().report_scalar(title=name, series=series, value=value, iteration=step)

    def log_video(self, name: str, **kwargs) -> None:
        # https://clear.ml/docs/latest/docs/guides/reporting/media_reporting/
        """Log video inputs to clearml.

        Args:
            name (str): The name of the video.
            video (Tensor): The video to be logged, expected to be in (T, C, H, W) format
                for consistency with other loggers.
            **kwargs: Other keyword arguments. By construction, log_video
                supports 'step' (integer indicating the step index) and 'fps' (default: 6).
        """
        self.video_log_counter += 1

        from clearml import Logger

        path = kwargs.pop("path", "./")

        Logger.current_logger().report_media(
            'video', name, iteration=1,
            local_path=path
        )

    def log_hparams(self, cfg: Union["DictConfig", Dict]) -> None:  # noqa: F821
        """Logs the hyperparameters of the experiment.

        Args:
            cfg (DictConfig or dict): The configuration of the experiment.
        """
        from omegaconf import OmegaConf

        if type(cfg) is not dict and _has_omegaconf:
            cfg = OmegaConf.to_container(cfg, resolve=True)

        self.task.connect(cfg)

    def __repr__(self) -> str:
        return f"ClearMLLogger(experiment={self.experiment.__repr__()})"

    def log_histogram(self, name: str, data: Sequence, **kwargs):
        # https://clear.ml/docs/latest/docs/references/sdk/logger/#report_histogram
        # https://clear.ml/docs/latest/docs/guides/reporting/scatter_hist_confusion_mat_reporting/

        # kwargs = {series = 'histogram series',
        #     "random histogram",
        #     iteration=0,
        # values=histogram,
        # xaxis="title x",
        # yaxis="title y",
        # labels = ['A', 'B'], # Optional
        # }

        from clearml import Logger

        Logger.current_logger().report_histogram(
            title=name,
            values=data,
            **kwargs
        )

    def close(self):
        self.task.close()
