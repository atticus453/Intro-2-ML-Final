import cv2
import torch
import os
import argparse
from datetime import datetime
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import torchvision.models as models

def main(weights_path, output_dir, conf_thres):
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device('cpu')
    device = torch.device('cpu')
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dog_emotion_clf = models.mobilenet_v3_large(pretrained=True).to(device)
    # dog_emotion_clf.classifier[3] = torch.nn.Sequential(
    #                         torch.nn.Linear(dog_emotion_clf.classifier[3].in_features, 1024).to(device),
    #                         torch.nn.Hardswish(),
    #                         torch.nn.Dropout(p=0.5),
    #                         torch.nn.Linear(1024, 512),
    #                         torch.nn.Hardswish(),
    #                         torch.nn.Dropout(p=0.5),
    #                         torch.nn.Linear(512, 4),
    #                         ).to(device)

    dog_emotion_clf = torch.load('./dog_emotion_weight.pth', map_location=device)
    dog_emotion_clf.eval()

    # 加載指定的 YOLOv5 模型權重，並設置置信度閾值
    print(f"Loading YOLO model from weights: {weights_path}")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
    yolo_model.conf = conf_thres  # 設置置信度閾值
    # 初始化下一個模型（示例：ResNet）
    next_model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    next_model.eval()

    # 打開攝像頭
    cap = cv2.VideoCapture(0)  # 0 表示默認攝像頭

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOv5 檢測
        results = yolo_model(frame)

        # 獲取框框結果
        for i, (*box, conf, cls) in enumerate(results.xyxy[0]):  # 每個檢測結果
            x1, y1, x2, y2 = map(int, box)  # 獲取框的坐標
            cropped = frame[y1:y2, x1:x2]  # 裁剪框框內容


            # 保存裁剪後的內容為 JPG 文件
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            crop_filename = os.path.join(output_dir, f"crop_{timestamp}_{i}.jpg")
            cv2.imwrite(crop_filename, cropped)
            from PIL import Image

            # 將 numpy.ndarray 轉換為 PIL 圖像
            cropped_pil = Image.fromarray(cropped)
            transformed_img_tensor = transform(cropped_pil)
            transformed_img_tensor = transformed_img_tensor.unsqueeze(0)
            outputs = dog_emotion_clf(transformed_img_tensor)
            predict = torch.max(outputs, 1)
            # # 將裁剪內容傳遞到下一個模型
            # resized = cv2.resize(cropped, (224, 224))  # 調整大小以適配下一個模型
            # input_tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0

            # with torch.no_grad():
            #     output = next_model(input_tensor)  # 下一個模型的輸出

            # 可選：在框框上顯示結果
            emotion = ["angry","happy","relaxed","sad"]
            emotion_color = {0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(255,255,255)}
            _,indices = predict
            result = int(indices[0])
            label = f"Class: {emotion[result]}, Conf: {conf:.2f}"
            emotion_color = {0:(255,0,0),1:(0,255,0),2:(0,0,255),3:(255,255,255)}
            cv2.rectangle(frame, (x1, y1), (x2, y2), emotion_color[result], 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, emotion_color[result], 2)
            # cv2.putText(frame, predict, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            print(predict)

        # 顯示檢測結果
        cv2.imshow('YOLO Real-Time Detection', frame)

        # 按 'q' 鍵退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 釋放資源
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # 使用 argparse 處理命令行參數
    parser = argparse.ArgumentParser(description="YOLO Real-Time Detection with Custom Output Directory, Weights, and Confidence Threshold")
    parser.add_argument('weights_path', type=str, help="Path to the YOLO weights file (e.g., best.pt)")
    parser.add_argument('output_dir', type=str, help="Path to the folder where cropped images will be saved")
    parser.add_argument('--conf', type=float, default=0.25, help="Confidence threshold for detection (default: 0.25)")
    args = parser.parse_args()

    # 執行主函數
    main(args.weights_path, args.output_dir, args.conf)
