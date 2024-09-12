from flax import nnx
import pickle


def save_training_data(data_path: str, data_filename: str, data: dict) -> None:
    """
    Save the data obtained by interacting with the environment.
    """
    with open(data_path + data_filename, "wb") as file:
        pickle.dump(data, file)
    print(f"Training data saved to {data_filename}")

def save_trained_modules(agent, ckpt_path: str, ckpt_filename: str) -> None:
    """
    Save the actor and critic network params.
    """
    training_state = {
        "actor_model": nnx.state(agent.actor),
        "critic_model": nnx.state(agent.critic),
    }
    with open(ckpt_path + ckpt_filename, "wb") as file:
        pickle.dump(training_state, file)
    print(f"Checkpoint saved to {ckpt_filename}")

def load_training_data(data_path: str, data_filename: str):
    """
    Load the training data.
    """
    with open(data_path + data_filename, "rb") as file:
        data = pickle.load(file)
    print(f"Checkpoint loaded from {data_filename}")

    return data

def load_trained_modules(ckpt_path: str, ckpt_filename: str):
    """
    Load the actor and critic network params.
    """
    with open(ckpt_path + ckpt_filename, "rb") as file:
        restored_state = pickle.load(file)
    print(f"Checkpoint loaded from {ckpt_filename}")

    return restored_state
