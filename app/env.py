from dotenv import find_dotenv, load_dotenv


def load_runtime_env() -> None:
    dotenv_path = find_dotenv(usecwd=True)
    if dotenv_path:
        load_dotenv(dotenv_path=dotenv_path)
