import torch
from Model import Model

def load_model(checkpoint_path):
    # Define model parameters
    input_size = 20
    hidden_size = 64
    output_size = 2

    # Initialize the model
    model = Model.Model(input_size, hidden_size, output_size)

    # Load the model state from checkpoint
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def preprocess_name(name):
    # Convert name to ASCII values and pad/truncate to fixed length
    name_transformed = [ord(char) for char in name]
    max_length = 20
    if len(name_transformed) > max_length:
        name_transformed = name_transformed[:max_length]
    else:
        name_transformed += [0] * (max_length - len(name_transformed))  # Pad with zeros
    return torch.tensor(name_transformed, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


def predict_gender(model, name):
    # Preprocess the name
    input_tensor = preprocess_name(name)

    # Pass through the model
    with torch.no_grad():
        model.eval()
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)

    # Interpret prediction
    gender = "MALE" if predicted.item() == 0 else "FEMALE"

    return gender


if __name__ == "__main__":
    # Load the trained model
    model = load_model("checkpoints/chk198.pth")  # Change to the path of your trained model

    # Get user input
    name = input("Enter a name: ")

    # Predict gender
    predicted_gender = predict_gender(model, name)

    # Print prediction
    print(f"The predicted gender of '{name}' is: {predicted_gender}")
