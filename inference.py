#!/usr/bin/env python3

import os
import shutil
import time
import json
import cv2
from dotenv import load_dotenv
from pyzbar.pyzbar import decode as pyzbar_decode
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import onnxruntime
import argparse
import requests
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from rapidocr_onnxruntime import RapidOCR
import pytesseract
from symspellpy.symspellpy import SymSpell
from pylibdmtx.pylibdmtx import decode

def draw_ocr_boxes(pil_image, ocr_data):
    """
    Draw bounding boxes on a Pillow image based on pytesseract OCR data.

    For each bounding box, draw a label in the top right corner. For level 5,
    the label is the recognized word (if available); for other levels, the label
    is the structural element name.

    Parameters:
      pil_image (PIL.Image): The image on which to draw.
      ocr_data (dict): The pytesseract output dictionary with keys such as:
          'level', 'left', 'top', 'width', 'height', 'text', etc.

    Returns:
      PIL.Image: The image with drawn bounding boxes and labels.
    """
    # Create a drawing context and a default font
    draw = ImageDraw.Draw(pil_image)

    # Map OCR level numbers to names.
    # Typically: 1: Page, 2: Block, 3: Paragraph, 4: Line, 5: Word.
    level_names = {1: "Page", 2: "Block", 3: "Paragraph", 4: "Line", 5: "Word"}

    # Number of detected elements
    n_elements = len(ocr_data['level'])

    for i in range(n_elements):
        level = ocr_data['level'][i]
        left = ocr_data['left'][i]
        top = ocr_data['top'][i]
        width = ocr_data['width'][i]
        height = ocr_data['height'][i]

        right = left + width
        bottom = top + height

        # For words (level 5), use the detected text if it exists.
        if level == 5:
            label = ocr_data['text'][i].strip()
            if not label:
                label = level_names.get(level, str(level))
        else:
            label = level_names.get(level, str(level))

        # Draw the bounding rectangle
        draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Determine text size to place the label in the top right corner of the box.
        font = ImageFont.load_default(size=20)
        text_height = 20
        text_width = font.getlength(label) + 2
        label_x = right - text_width  # align text's right edge with the box's right edge
        label_y = top

        # Optionally draw a filled background for the text to ensure readability.
        draw.rectangle([label_x, label_y, label_x + text_width, label_y + text_height], fill="red")
        draw.text((label_x, label_y), label, fill="white", font=font)

    return pil_image

def convert_bbox(corners):
  """
  Convert a list of corner coordinates to x1, y1, x2, y2 format.
  :param corners: A list of 4 corner coordinates, each a [x, y] list.
  :return: A tuple (x1, y1, x2, y2)
  """
  xs = [c[0] for c in corners]
  ys = [c[1] for c in corners]
  x1 = int(min(xs))
  y1 = int(min(ys))
  x2 = int(max(xs))
  y2 = int(max(ys))
  return (x1, y1, x2, y2)

def threshold_img(img, c):
  #img = img.resize((img.width*2,img.height*2), resample=Image.BICUBIC)
  img = np.array(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2), interpolation=cv2.INTER_LINEAR)
  #img = cv2.medianBlur(img, 3)  # Apply median blur for reducing noise
  #img = cv2.GaussianBlur(img, (3,3), 0)
  img = cv2.adaptiveThreshold(
    img,
    maxValue=255,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # or ADAPTIVE_THRESH_MEAN_C
    thresholdType=cv2.THRESH_BINARY,                # or THRESH_BINARY_INV
    blockSize=31,    # Size of the pixel neighborhood (must be odd and >1)
    C=c              # Constant subtracted from the mean or weighted mean
  )
  #img = cv2.resize(img, (int(img.shape[1]/2),int(img.shape[0]/2)))
  #kernel = np.ones((3,3), np.uint8)
  #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
  #img = cv2.Canny(img, 100, 200)
  #img = cv2.dilate(img, np.ones((5,5), np.uint8), iterations=1)
  #img = cv2.GaussianBlur(img, (3,3), 0)
  #img = cv2.medianBlur(img, 3)  # Apply median blur for reducing noise
  #ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  return Image.fromarray(img, mode='L')

def get_label_angle(img, canny_thresh1=100, canny_thresh2=150, debug=False):
  img = np.array(img)
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, canny_thresh1, canny_thresh2, apertureSize=3)
  lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)
  cv2.imwrite('debug/edges.jpg', edges)
  if lines is None:
      return None, []

  # Extract angles in degrees
  angles = []
  for rho_theta in lines:
      rho, theta = rho_theta[0]
      angle_deg = np.rad2deg(theta) % 180  # Normalize to [0, 180)
      if angle_deg > 20 and angle_deg < 120:
        angles.append(angle_deg)
  #bins = [0]*19
  #for v in angles:
  #  index = min(int(v) // 10, 18)
  #  bins[index] += 1
  #for i, count in enumerate(bins):
  #  start = i * 10
  #  end = start + 9 if i < 18 else 180
  #  label = f"{start:03d}-{end:03d}"
  #  bar = '█' * count
  #  print(f"{label}: {bar}")
  if len(angles) < 3:
    print("ANGLE ADJUSTMENT FAILED")
    print(angles)
    return 0
  else:
    angle_adjustment = int(sum(angles)/len(angles)) - 90
    print(angle_adjustment)
    return angle_adjustment

def barcode_reader(img):
  width, height = img.size
  img = img.crop((int(width * 0.2), int(height * 0.7), width - int(width * 0.2), height - int(height * 0.05)))
  img = np.array(img)
  img = cv2.resize(img, (img.shape[1]*2,img.shape[0]*2), interpolation=cv2.INTER_LINEAR)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  #img = cv2.medianBlur(img, 3)  # Apply median blur for reducing noise
  ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
  img = Image.fromarray(img, mode='L')
  img.save('barcode.jpg')
  results = pyzbar_decode(img)
  return results

def qr_reader(img):
  img = np.array(img)
  img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  img = cv2.medianBlur(img, 3)  # Apply median blur for reducing noise
  ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  img = Image.fromarray(img, mode='L')
  img.save('debug/qr.jpg')
  data = decode(img, max_count=1)
  if data:
    label = str(data[0].data.decode('utf-8'))
    return label
  else:
    return None

def raw_ocr(img, method='rapidocr'):
  start = time.time()

  if method == 'tesseract':
    results = pytesseract.image_to_string(img, lang='eng', config='--oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-(); --tessdata-dir ./tessdata').lower().split('\n') # config='tessedit_do_invert=0'
    #data = pytesseract.image_to_data(img, lang='eng', output_type=pytesseract.Output.DICT)
    #tesseract_img = draw_ocr_boxes(img, data)
    #tesseract_img.save('tesseract_img.jpg')
    results_corrected = ' '.join([sym_spell.lookup_compound(result, max_edit_distance=2)[0].term for result in results if len(result) > 2]).split(' ')
    if args.verbose:
      print(f"Tesseract raw result: {' '.join(results)}")
      print(f"Tesseract corrected result: {' '.join(results_corrected)}")

  elif method == 'rapidocr':
    results, _ = rapidocr_engine(img, use_cls=False, use_det=True)
    if not results:
      print("RapidOCR failed")
      return 'None'
    results = [block[1].lower() for block in results]
    results_corrected = ' '.join([sym_spell.word_segmentation(result).corrected_string for result in results if len(result) > 2]).split(' ')
    if args.verbose:
      print(f"RapidOCR raw result: {' '.join(results)}")
      print(f"RapidOCR corrected result: {' '.join(results_corrected)}")

  elif method == 'gpt':
    img.thumbnail((480,480))
    results = ask_gpt("You are an OCR system. Only respond with the answer.", f"Extract all the text in the box. The text is very likely to be obscured or smudged so try your best to guess the correct text.", img)
    results = results.lower().split('\n')
    results_corrected = ' '.join([sym_spell.word_segmentation(result).corrected_string for result in results if len(result) > 2]).split(' ')
    if args.verbose:
      print(f"GPT raw result: {' '.join(results)}")
      print(f"GPT corrected result: {results_corrected}")
  else:
    result = None

  with open('dict.txt', 'r') as f:
    matches = f.read().lower().split('\n')
  results_corrected = ' '.join([match.replace("'","_") for match in results_corrected if match in matches and match != '']).replace('blanking plate','blanking_plate')
  result = results_corrected.replace('medium','pick_up')
  if args.profile:
    print('OCR time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")

  return result

def ocr(img):
  # First try
  img_thresh1 = threshold_img(img, 20)
  img_thresh1.save('debug/label_thresh1.jpg')
  text = raw_ocr(img_thresh1, 'tesseract')
  if args.verbose:
    print(f"OCR 1: {text}")
  matches = text.split(' ')
  if len(matches) < 3:
    # Second try
    img_thresh2 = threshold_img(img, 10)
    img_thresh2.save('debug/label_thresh2.jpg')
    text = raw_ocr(img_thresh2, 'tesseract')
    if args.verbose:
      print(f"OCR 1.2: {text}")
    matches = text.split(' ')
    if len(matches) < 3:
      # Third try
      text = raw_ocr(img, 'rapidocr')
      if args.verbose:
        print(f"OCR 2: {text}")
      matches = text.split(' ')
      if len(matches) < 3:
        # Fourth try
        text = raw_ocr(img, 'gpt')
        if args.verbose:
          print(f"OCR 3: {text}")
        matches = text.split(' ')
        if len(matches) < 3:
          print("Could not extract fittings - exiting...")
          return None

  while matches[0] == 'plug':
    matches.pop(0)

  results = {
    'chassis': 'None',
    #'chassis': chassis,
    'tank': 'None',
    #'tank': tank,
    'A': matches[0].lower().strip() if len(matches) > 0 else None,
    'B': matches[1].lower().strip() if len(matches) > 1 else None,
    'C': matches[2].lower().strip() if len(matches) > 2 else None
  }
  return results

# Function to encode the image
def encode_image(image):
  #image.thumbnail((128,128))
  image.save('./pil_img.jpg')
  with open('./pil_img.jpg', "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def ask_gpt(sys, msg, img=None):
  # Getting the base64 string
  headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
  }
  if img:
    b64_img = encode_image(img)
    content = [
      { "type": "text", "text": msg },
      { "type": "image_url", "image_url": { "url": f"data:image/jpeg;base64,{b64_img}", "detail": "low" } }
    ]
  else:
    content = [
      { "type": "text", "text": msg }
    ]
  payload = {
    "model": "gpt-4o-mini",
    #"model": "gpt-3.5-turbo",
    #"model": "o1-mini",
    "messages": [
      { "role": "system", "content": [ { "type": "text", "text": sys }] },
      { "role": "user", "content": content }
    ],
    "max_tokens": 300
  }
  response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()
  if 'choices' in response.keys():
    result = response['choices'][0]['message']['content']
  else:
    print(response)
    result = None
  return result

def draw_label(draw, box, text, color):
  colors = [
    (255, 0, 0),   # Red
    (0, 255, 0),   # Green
    (0, 0, 255),   # Blue
    (255, 255, 0), # Yellow
    (255, 0, 255), # Magenta
    (0, 255, 255), # Cyan
  ]
  size = font.getlength(text)
  box_width = size + 2
  box_height = 30
  box_position = (box[0], box[1] - box_height)
  draw.rectangle(box, outline=colors[color], width=5)
  draw.rectangle([box_position, (box_position[0] + box_width, box_position[1] + box_height)], fill=colors[color])
  draw.text((box_position[0], box_position[1]-4), text, fill=colors[color+1], font=font)

def contains(object, region):
  if (object[0] > region[0]) and (object[1] > region[1]) and (object[2] < region[2]) and (object[3] < region[3]):
    return True
  else:
    return False

def center_contains(object, region):
  if (object[0] > region[0]) and (object[1] > region[1]) and (object[0] < region[2]) and (object[1] < region[3]):
    return True
  else:
    return False

def filter_det(result, classes, conf):
  indices = {v: k for k, v in result.names.items()}
  return [obj for obj in result.boxes if obj.cls[0] in [indices[c] for c in classes] and obj.conf > conf]

def allowed_file(filename):
  return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Calculate the cross product to determine the side
def find_side(p1, p2, p):
  return (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])

def softmax(x):
    x = x.reshape(-1)
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def convert_bbox_wh(bbox):
  return [ int(bbox[0]-bbox[2]/2), int(bbox[1]-bbox[3]/2), int(bbox[0]+bbox[2]/2), int(bbox[1]+bbox[3]/2) ]

def preprocess(image, imgsz):
    #img = image.copy()
    #img.thumbnail((imgsz,imgsz))
    #bg1 = Image.new('RGB', (imgsz, imgsz), (0, 0, 0))
    #offset = ((imgsz - img.width) // 2, (imgsz - img.height) // 2)
    #bg1.paste(img, offset)

    img_resize = image.copy()
    img_resize.thumbnail((imgsz*2,imgsz*2))
    bg2 = Image.new('RGB', (imgsz*2, imgsz*2), (0, 0, 0))
    offset = ((imgsz*2 - img_resize.width) // 2, (imgsz*2 - img_resize.height) // 2)
    bg2.paste(img_resize, offset)

    bg1 = bg2.resize((imgsz,imgsz))

    image_data = np.array(bg1).transpose(2, 0, 1) / 255.0
    # convert the input data into the float32 input
    img_data = image_data.astype('float32')

    #normalize
    #mean_vec = np.array([0.485, 0.456, 0.406])
    #stddev_vec = np.array([0.229, 0.224, 0.225])
    #norm_img_data = np.zeros(img_data.shape).astype('float32')
    #for i in range(img_data.shape[0]):
    #    norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]

    #add batch channel
    norm_img_data = img_data.reshape(1, 3, imgsz, imgsz).astype('float32')
    return norm_img_data, bg1, bg2

def postprocess(output, conf, iou):
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    boxes = []
    scores = []
    class_ids = []
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= conf:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([x, y, w, h])
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf, iou)
    boxes = list(zip(class_ids, scores, boxes))
    boxes = [boxes[i] for i in indices]
    return boxes

def process(files):
  imgs_original = [Image.open(img) for img in files]
  #shutil.rmtree("output")
  #os.mkdir("./output")
  classes = [ 'plug', 'sender', 'jk_sender', 'clamp', 'hose', 'elbow', 'allen', 'blanking', 'label', 'pickup', 'breather' ]
  imgsz = 960
  input_name = onnx_session.get_inputs()[0].name
  for img_indx, img_original in enumerate(imgs_original):
    start = time.time()
    input_data, img, img_double = preprocess(img_original, imgsz=imgsz)
    if args.profile:
      print('Preprocess time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    start = time.time()
    raw_result = onnx_session.run([], {input_name: input_data})
    if args.profile:
      print('Inference time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    start = time.time()
    results = postprocess(raw_result, args.conf, args.iou)
    if args.profile:
      print('Postprocess time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    start = time.time()
    if args.debug:
      img_draw = img_double.copy()
      draw = ImageDraw.Draw(img_draw)

    div_1 = None
    div_2 = None
    for result in results:
      if classes[result[0]] in ['sender', 'jk_sender', 'blanking']:
        div_1 = [coord/imgsz for coord in result[2][:2]]
      if classes[result[0]] in ['breather']:
        div_2 = [coord/imgsz for coord in result[2][:2]]
    if div_1 and div_2:
      if div_1[1] < div_2[1]:
        upright = True
      else:
        upright = False
    elif div_1:
      print('Breather not found, assuming upright...')
      upright = True # default
    elif div_2:
      print('Plate not found, assuming upright...')
      upright = True # default
    else:
      print('Plate and Breather not found, assuming upright...')
      upright = True # default

    label = None


    for result in results:
      if classes[result[0]] == 'label':
        x1n, y1n, x2n, y2n = [ x/imgsz for x in convert_bbox_wh(result[2])]
        crop = ( int(x1n*imgsz*2), int(y1n*imgsz*2), int(x2n*imgsz*2), int(y2n*imgsz*2) )
        label_img = img_double.crop(crop).rotate(0 if upright else 180)
        width, height = label_img.size
        label_crop = label_img.crop((int(width * 0.1), int(height * 0.55), width - int(width * 0.1), height - int(height * 0.15)))
        label_crop = label_crop.rotate(get_label_angle(label_crop))
        start = time.time()
        qr_crop = label_img.crop((int(width * 0), int(height * 0.25), width - int(width * 0.2), height - int(height * 0.3)))
        tank = qr_reader(qr_crop)
        #results = barcode_reader(label_img)
        if args.profile:
          print('barcode time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
        if args.debug:
          print("Saving label img to label.jpg")
          label_img.save('debug/label.jpg')
        label = ocr(label_crop)
        if label:
          label['tank'] = tank
        break
    if not label:
      if os.path.exists('label.json'):
        if args.verbose:
          print('Reading label ocr data from label.json')
        with open('label.json', 'r') as f:
          label = json.load(f)
        os.remove('label.json')
      else:
        print('Label not found or unreadable, exiting...')
        return None
    else:
      with open('label.json', 'w') as f:
        # Cache label ocr
        f.write(json.dumps(label))
      if args.debug:
        draw_label(draw, [coord*2 for coord in convert_bbox_wh(result[2])], ', '.join([label['A'] if label['A'] else 'None', label['B'] if label['B'] else 'None', label['C'] if label['C'] else 'None']), 2)

    pos_a = None
    pos_b = None
    pos_c = None
    match_txt = [ "WRONG", "CORRECT" ]
    matches = []
    for result in results:
      if classes[result[0]] in ['sender', 'jk_sender', 'blanking']:
        pos_a = classes[result[0]]
        matches.append(pos_a in label['A'])
        if args.debug:
          draw_label(draw, [coord*2 for coord in convert_bbox_wh(result[2])], match_txt[matches[-1]], matches[-1])
      if classes[result[0]] in ['plug', 'elbow', 'allen']:
        side = find_side(div_1, div_2, [coord/imgsz for coord in result[2][:2]])
        if side > 0: # LEFT
          pos_b = classes[result[0]]
          matches.append(pos_b in label['B'].replace('return', 'elbow').replace('pick', 'elbow'))
          if args.debug:
            draw_label(draw, [coord*2 for coord in convert_bbox_wh(result[2])], match_txt[matches[-1]], matches[-1])
        elif side < 0: # RIGHT
          pos_c = classes[result[0]]
          matches.append(pos_c in label['C'].replace('return', 'elbow').replace('pick', 'elbow'))
          if args.debug:
            draw_label(draw, [coord*2 for coord in convert_bbox_wh(result[2])], match_txt[matches[-1]], matches[-1])

    if args.debug:
      index = len(os.listdir('./output'))
      #img_draw.save(f"./output/{files[img_indx].split('/')[-1].replace('.png','.jpg')}")
      img_draw.save(f"./output/image_{index}.jpg")

    if (pos_a==None and pos_b==None and pos_c==None):
      return 'label'
    if not (pos_a and pos_b and pos_c):
      print('Some fittings not detected, exiting...')
      return None

    result = f"Chassis: {label['chassis']}\n"
    result += f"Tank: {label['tank']}\n"
    result += f"Fittings: {label['A']}, {label['B']}, {label['C']}\n"
    result += f"Found: {pos_a}, {pos_b}, {pos_c}\n"
    result += ', '.join([match_txt[match] for match in matches]) + '\n'
    result += 'ACCEPTED' if (all(matches) and len(matches)==3) else 'REJECTED'
    if args.profile:
      print('Logic time: ' + str(np.round((time.time() - start) * 1000, 2)) + " ms")
    return result

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
shutil.rmtree(UPLOAD_FOLDER)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
font = ImageFont.load_default(size=30)

@app.route('/upload', methods=['POST'])
def upload_file():
  if args.profile:
    start_time = time.time()
  if 'file' not in request.files:
    return jsonify({'message': 'No file part'}), 400
  file = request.files['file']
  if file.filename == '':
    return jsonify({'message': 'No selected file'}), 400
  if file and allowed_file(file.filename):
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    result = process([filepath])
    if args.debug:
      print(result)
    if args.profile:
      print(f"Total time: {time.time()-start_time}")
    if result:
      return jsonify({'message': result}), 200
    else:
      return jsonify({'message': "Failed"}), 400
  else:
    return jsonify({'message': 'Invalid file type'}), 400


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-i', '--imgs', nargs='+')
  parser.add_argument('-m', '--model', default='models/model_33.onnx')
  parser.add_argument('-s', '--serve', action='store_true')
  parser.add_argument('-p', '--profile', action='store_true')
  parser.add_argument('-v', '--verbose', action='store_true')
  parser.add_argument('-d', '--debug', action='store_true')
  parser.add_argument('-o', '--ocr', action='store_true')
  parser.add_argument('-c', '--conf', type=float, default=0.4)
  parser.add_argument('-u', '--iou', type=float, default=0.5)
  args = parser.parse_args()

  # OpenAI API Key
  load_dotenv()
  api_key = os.getenv("OPENAI_API_KEY")
  rapidocr_engine = RapidOCR(rec_model_path='./models/en_PP-OCRv3_rec_infer.onnx', det_model_path='./models/en_PP-OCRv3_det_infer.onnx')
  onnx_session = onnxruntime.InferenceSession(args.model, None)

  sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
  sym_spell.create_dictionary('dict.txt')

  if args.serve:
    app.run(debug=False, host='0.0.0.0', port=5002)
  elif args.imgs:
    if args.ocr:
      for img in [Image.open(img) for img in args.imgs]:
        result = ocr(img)
        print(f"Final OCR result: {result}")
    else:
      result = process(args.imgs)
      print(result)
