import typing
import os
task_name = 'botevade'


def set_task(task: str):
    global task_name
    task_name = task


def data_folder():
    folder = os.path.join("data", task_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def buffers_folder():
    folder = os.path.join("buffers", task_name)
    os.makedirs(folder, exist_ok=True)
    return folder


def models_folder():
    return os.path.join("models", task_name)


def data_file(model_name: str, run_identifier: str) -> str:
    return os.path.join(data_folder(), f"{model_name}_{run_identifier}.zip")


def out_buffer_file(model_name: str, run_identifier: str) -> str:
    return os.path.join(buffers_folder(), f"{model_name}_{run_identifier}_buffer.pickle")


def in_buffer_file(replay_buffer_file: str) -> str:
    return os.path.join(buffers_folder(), replay_buffer_file)


def model_config_file(model_name: str) -> str:
    return os.path.join(models_folder(), f"{model_name}_config.json")


def tensor_board_logs_folder(model_name: str, run_identifier: str) -> str:
    folder = os.path.join("tensorboard_logs", task_name, f"{model_name}_{run_identifier}")
    os.makedirs(folder, exist_ok=True)
    return folder


def video_folder(model_name: str, run_identifier: str) -> str:
    folder = os.path.join("videos", task_name, model_name, run_identifier)
    os.makedirs(folder, exist_ok=True)
    return folder


def run_identifier(run_id: typing.Optional[str]) -> str:
    if run_id:
        return run_id
    else:
        from datetime import datetime
        current_datetime = datetime.now()
        return current_datetime.strftime("%Y%m%d_%H%M%S")
