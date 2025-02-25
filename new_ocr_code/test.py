import torch
import  os
from PIL import Image
from torchvision import transforms
from model.attention_ocr import OCR_Model
from src.tokenizer import Tokenizer
import cv2 , numpy as np

img_width = 160
img_height = 60

#img_width = 400
#img_height = 200

max_len = 10
nh = 512
device = 'cpu'
chars = list('1234567890ABCDEFGHIJKLMNOPQRSTUVWXYZ')
n_chars = 11
test_images_path = "/var/data/project/new_ocr/pix2pix/new_dir"
#test_images_path = "/var/data/project/new_ocr/Lplate_org_data/DL_images/"

output_path = '/var/data/project/new_ocr/Lplate_org_data/reslut/'




tokenizer = Tokenizer(chars)
model = OCR_Model(img_width, img_height, nh, tokenizer.n_token,max_len + 1, tokenizer.SOS_token, tokenizer.EOS_token).to(device=device)
model.load_state_dict(torch.load('/home/synlabs/project/light_weight_ocr/new_ocr_code/chkpoint/time_2021-03-12_04-34-45_epoch_15.pth'))

img_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485], std=(0.229)),])

model.eval()
for image_name in os.listdir(test_images_path):
    image_path = os.path.join(test_images_path , image_name)
    image = Image.open(image_path)
    image = image.convert("L")
    image_1 = image.resize((400, 200))
    image = image.resize((img_width, img_height))
    d = img_trans(image)
    with torch.no_grad():
        pred = model(d.unsqueeze(0))
    rst = tokenizer.translate(pred.squeeze(0).argmax(1))
    image_1 = np.array(image_1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    image_1 = cv2.cvtColor(image_1, cv2.COLOR_RGB2BGR)
    cv2.putText(image_1, rst, (10, 50), font, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imwrite(output_path + image_name, image_1)
    print(rst)


    
 # OOnx convert code :
import torch
import cv2
from torchvision import transforms
img_trans = transforms.Compose([transforms.ToTensor()])
model=torch.load("lpr.pth", map_location=torch.device('cpu'))
model.eval()
width_height = (160, 60)
image = cv2.imread("img.jpg")
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = cv2.resize(image, width_height)
d = img_trans(image)
d = d.unsqueeze(0)
with torch.no_grad():
    torch_out = model(d)
torch.onnx.export(model,  # model being run
                  d,  # model input (or a tuple for multiple inputs)
                  "lpr.onnx",  # where to save the model (can be a file or file-like object)
                  export_params=True,  # store the trained parameter weights inside the model file
                  opset_version=10,  # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],  # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                'output': {0: 'batch_size'}})


