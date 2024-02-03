import torch
import torch.nn as nn
import cv2
from torchvision import transforms
from torchvision.transforms import ToPILImage
import time
import sys

Device =torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
classes = ['a','b','c','d','del','e','f','g','h','i','j','k','l','m','n','nothing','o','p','q','r','s','space','t','u','v','w','x','y','z']
# model_path = "model/asl_alphabet_model_resized.pt"
model_path = "model/asl_alphabet_model_resized (2).pt"



class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 25 * 25, 128)
        self.fc2 = nn.Linear(128, len(classes))

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = x.view(-1, 25 * 25 * 64)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNNModel().to(Device)
# if os.path.exists(model_path):
model.load_state_dict(torch.load(model_path,map_location=torch.device(Device)))
print("model loaded from ",model_path)


# Load model
model.eval()  # Set model to evaluation mode

new_transform = transforms.Compose([
    transforms.Resize((100, 100)),
    transforms.Grayscale(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Open the camera
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Error reading frame from camera")
        break

    # Preprocess the frame (if necessary)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #frame_tensor = torch.tensor(frame_rgb).permute(2, 0, 1).float() / 255.0  # Normalize to [0, 1]

   # Pass the frame to the model
    with torch.no_grad():
        pli_image = ToPILImage()(frame_rgb)
        transformed = new_transform(pli_image)
        output = model(transformed)  # Add batch dimension
        
    _, prediction = torch.max(output.data, 1)
    pred_in_text = classes[prediction[0]]
    
    # Write prediction on the frame
    cv2.putText(frame, f"Prediction: {pred_in_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 70, 30), 2)
    # cv2.putText(frame, f"Confidence: {_[0]:.2f}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 70, 30), 2)

    # Display the frame (optional)
    cv2.imshow("Camera", frame)
    # Exit on 'q' key press
    if cv2.waitKey(1) == ord("q") or cv2.getWindowProperty("Camera", 0) < 0:
        break
    time.sleep(0.1)

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
