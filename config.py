import typing
import os


task_name = ""
model_name = ""
run_identifier = ""


def set_run_identifier(run_id: typing.Optional[str]):
    global run_identifier

    if run_id:
        run_identifier = run_id
    else:
        from datetime import datetime
        current_datetime = datetime.now()
        run_identifier = current_datetime.strftime("%Y%m%d%H%M%S")

def set_task(task: str):
    global task_name
    task_name = task


def set_model(model: str):
    global model_name
    model_name = model


def data_folder():
    folder = os.path.join("data", task_name, model_name, run_identifier)
    os.makedirs(folder, exist_ok=True)
    return folder


def buffers_folder():
    folder = os.path.join("buffers", task_name, model_name, run_identifier)
    os.makedirs(folder, exist_ok=True)
    return folder


def experiments_folder():
    folder = os.path.join("experiments", task_name, model_name, run_identifier)
    os.makedirs(folder, exist_ok=True)
    return folder


def experiments_name():
    from datetime import datetime
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{now}_{task_name}_{model_name}_{run_identifier}"
    return name


def models_folder():
    return os.path.join("models", task_name)


def data_file(suffix: str = "", cycle: str = "") -> str:
    if cycle:
        if suffix:
            return os.path.join(data_folder(), f"{run_identifier}_{suffix}_{cycle}.zip")
        else:
            return os.path.join(data_folder(), f"{run_identifier}_{cycle}.zip")
    else:
        if suffix:
            return os.path.join(data_folder(), f"{run_identifier}_{suffix}.zip")
        else:
            return os.path.join(data_folder(), f"{run_identifier}.zip")



def performance_file(suffix: str = "") -> str:
    if suffix:
        return os.path.join(data_folder(), f"performance_{suffix}.json")
    else:
        return os.path.join(data_folder(), f"performance.json")


def out_buffer_file(suffix: str = "") -> str:
    if suffix:
        return os.path.join(buffers_folder(), f"buffer_{suffix}.pickle")
    else:
        return os.path.join(buffers_folder(), f"buffer.pickle")


def in_buffer_file(replay_buffer_file: str) -> str:
    return os.path.join(buffers_folder(), replay_buffer_file)


def model_config_file() -> str:
    return os.path.join(models_folder(), f"{model_name}_config.json")


def tensor_board_logs_folder(suffix: str = "") -> str:
    if suffix:
        folder = os.path.join("tensorboard_logs", task_name, model_name, run_identifier, suffix)
    else:
        folder = os.path.join("tensorboard_logs", task_name, model_name, run_identifier)
    os.makedirs(folder, exist_ok=True)
    return folder


def video_folder() -> str:
    folder = os.path.join("videos", task_name, model_name, run_identifier)
    os.makedirs(folder, exist_ok=True)
    return folder


if __name__ == "__main__":
    set_task("TEST_TASK")
    set_model("TEST_MODEL")
    set_run_identifier("RUNID")
    print("Data Folder:", data_folder())
    assert (data_folder() == os.path.join("data", "TEST_TASK", "TEST_MODEL", "RUNID"))
    print("Tensor Board Folder", tensor_board_logs_folder())
    assert (tensor_board_logs_folder() == os.path.join("tensorboard_logs", "TEST_TASK", "TEST_MODEL_RUNID"))
