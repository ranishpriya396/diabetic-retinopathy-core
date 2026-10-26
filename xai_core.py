import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import timm

class RetinalXAI:
    def __init__(self, model):
        """
        Stage 9: Native Explainable AI Core. Uses PyTorch full backward hooks 
        to extract layer gradients completely offline.
        """
        self.model = model
        # Target the deepest feature module block of the ResNet-50 architecture
        self.target_layer = self.model.layer4[-1]
        
        self.gradients = None
        self.activations = None
        
        # Register hooks to catch forward and backward gradient flows cleanly
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        # Full backward hooks return a tuple of gradients; extract the output gradient map
        self.gradients = grad_output[0]

    def generate_heatmap(self, cv_img):
        """
        Calculates feature weights using native backpropagation and blends 
        a standard diagnostic colormap overlay over the retina image.
        """
        input_tensor = self.transform(cv_img).unsqueeze(0)
        
        # 1. Run forward pass through network nodes
        self.model.zero_grad()
        output = self.model(input_tensor)
        
        # Target the class driving the highest probability value
        target_class = torch.argmax(output)
        
        # 2. Trigger backpropagation to compute raw feature gradients
        output[0, target_class].backward()
        
        # 3. Pull tracked gradients and activations from memory stacks
        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # 4. Compute spatial global average pooling weights
        weights = np.mean(gradients, axis=(1, 2))
        
        # 5. Build weighted spatial activation matrix map
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]
            
        # 6. Apply ReLU threshold constraints and normalize array bounds
        cam = np.maximum(cam, 0)
        if np.max(cam) != 0:
            cam = cam / np.max(cam)
            
        # 7. Resize matrix to fit our original OpenCV viewing size dimensions
        cam = cv2.resize(cam, (cv_img.shape[1], cv_img.shape[0]))
        
        # Convert map into standard clinical red-and-blue overlay
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        cam_image = cv2.addWeighted(cv_img, 0.7, heatmap, 0.3, 0)
        
        return cam_image

# --- XAI Verification Test Routine ---
if __name__ == "__main__":
    test_retina = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(test_retina, (200, 200), 180, (20, 40, 140), -1) 
    cv2.circle(test_retina, (250, 230), 15, (10, 210, 235), -1) 
    
    print("⏳ Synchronizing model arrays with Native XAI hooks...")
    stable_model = timm.create_model('resnet50.a1_in1k', pretrained=True)
    stable_model.eval()
    
    xai_engine = RetinalXAI(stable_model)
    heatmap_output = xai_engine.generate_heatmap(test_retina)
    
    cv2.imwrite("xai_heatmap_test.jpg", heatmap_output)
    print("✅ Success! Native Grad-CAM executed perfectly. Saved to 'xai_heatmap_test.jpg'")
