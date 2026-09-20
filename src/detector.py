# src/detector.py — Multi-Object Localization, Crop Classification & Visual Annotation
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

def find_candidate_regions(pil_img, min_area_ratio=0.012, max_area_ratio=0.94):
    img_rgb = np.array(pil_img.convert('RGB'))
    h_img, w_img, _ = img_rgb.shape
    total_area = h_img * w_img
    
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    
    s_channel = img_hsv[:, :, 1]
    v_channel = img_hsv[:, :, 2]
    l_channel = img_lab[:, :, 0]
    
    is_white_bg = (l_channel > 215) & (s_channel < 45)
    is_black_bg = (v_channel < 25)
    fg_mask = ~(is_white_bg | is_black_bg)
    
    mask_u8 = (fg_mask.astype(np.uint8)) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_clean = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    min_area = total_area * min_area_ratio
    max_area = total_area * max_area_ratio
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect = w / float(h)
            if 0.15 < aspect < 6.0:
                boxes.append([x, y, x + w, y + h, area])
                
    if not boxes or (len(boxes) == 1 and boxes[0][4] > total_area * 0.88):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edged = cv2.Canny(blurred, 30, 150)
        edged = cv2.dilate(edged, kernel, iterations=2)
        contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if min_area <= area <= max_area:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect = w / float(h)
                if 0.2 < aspect < 5.0:
                    boxes.append([x, y, x + w, y + h, area])

    if not boxes:
        return [(0, 0, w_img, h_img)]

    boxes = np.array(boxes)
    pick = []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = (x2 - x1) * (y2 - y1)
    idxs = np.argsort(area)[::-1]

    overlap_thresh = 0.35
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[idxs[1:]])
        yy1 = np.maximum(y1[i], y1[idxs[1:]])
        xx2 = np.minimum(x2[i], x2[idxs[1:]])
        yy2 = np.minimum(y2[i], y2[idxs[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        overlap = (w * h) / area[idxs[1:]]

        idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))

    selected_boxes = []
    for idx in pick:
        bx1, by1, bx2, by2 = int(x1[idx]), int(y1[idx]), int(x2[idx]), int(y2[idx])
        selected_boxes.append((bx1, by1, bx2 - bx1, by2 - by1))

    selected_boxes = sorted(selected_boxes, key=lambda b: (b[1] // 50, b[0]))
    return selected_boxes


def extract_crop_with_padding(pil_img, box, padding_ratio=0.08):
    w_img, h_img = pil_img.size
    x, y, w, h = box
    
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w_img, x + w + pad_x)
    y2 = min(h_img, y + h + pad_y)
    
    return pil_img.crop((x1, y1, x2, y2))


def draw_bounding_boxes(pil_img, item_results):
    annotated = pil_img.copy().convert('RGBA')
    draw = ImageDraw.Draw(annotated)
    w_img, h_img = annotated.size
    
    thickness = max(3, int(min(w_img, h_img) / 160))
    
    for item in item_results:
        idx = item['index']
        box = item['box']
        safe = item['verdict_info']['safe']
        status = item['verdict_info']['status']
        conf = item['confidence']
        
        x, y, w, h = box
        x1, y1, x2, y2 = x, y, x + w, y + h
        
        box_color = (46, 204, 113, 255) if safe else (231, 76, 60, 255)
        badge_bg = (39, 174, 96, 230) if safe else (192, 57, 43, 230)
        
        for t in range(thickness):
            draw.rectangle([x1 - t, y1 - t, x2 + t, y2 + t], outline=box_color)
            
        label_text = f'#{idx}: {status} ({conf:.1f}%)'
        
        font_size = max(13, int(min(w_img, h_img) / 36))
        try:
            font = ImageFont.truetype('arial.ttf', font_size)
        except Exception:
            font = ImageFont.load_default()
            
        text_bbox = draw.textbbox((x1, y1), label_text, font=font)
        tb_w = text_bbox[2] - text_bbox[0] + 16
        tb_h = text_bbox[3] - text_bbox[1] + 10
        
        badge_y1 = max(0, y1 - tb_h - 4)
        badge_y2 = badge_y1 + tb_h
        badge_x2 = min(w_img, x1 + tb_w)
        
        draw.rectangle([x1, badge_y1, badge_x2, badge_y2], fill=badge_bg)
        draw.text((x1 + 8, badge_y1 + 4), label_text, fill=(255, 255, 255, 255), font=font)

    return annotated.convert('RGB')


def analyze_image_multiobject(pil_img, model, labels, get_verdict_func, img_size=224):
    boxes = find_candidate_regions(pil_img)
    is_multi = len(boxes) > 1
    
    item_results = []
    
    for i, box in enumerate(boxes, start=1):
        crop = extract_crop_with_padding(pil_img, box)
        crop_resized = crop.resize((img_size, img_size))
        
        arr = np.expand_dims(np.array(crop_resized, dtype=np.float32) / 255.0, axis=0)
        preds = model.predict(arr, verbose=0)[0]
        
        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx]) * 100.0
        cls_name = labels[top_idx]
        v_info = get_verdict_func(cls_name)
        
        top3_indices = np.argsort(preds)[::-1][:3]
        top3 = [
            {'class': labels[idx], 'confidence': float(preds[idx]) * 100.0, 'status': get_verdict_func(labels[idx])['status']}
            for idx in top3_indices
        ]
        
        item_results.append({
            'index': i,
            'box': box,
            'crop': crop,
            'class_name': cls_name,
            'confidence': conf,
            'verdict_info': v_info,
            'top3': top3
        })
        
    annotated_img = draw_bounding_boxes(pil_img, item_results)
    
    total_items = len(item_results)
    fresh_count = sum(1 for item in item_results if item['verdict_info']['safe'])
    rotten_count = total_items - fresh_count
    
    if rotten_count == 0:
        batch_verdict = 'All items appear fresh and safe to consume.'
        batch_safe = True
    elif fresh_count == 0:
        batch_verdict = 'All detected items exhibit signs of rot, decay, or disease — do not consume.'
        batch_safe = False
    else:
        batch_verdict = f'Mixed Batch: {fresh_count} item(s) appear fresh, but {rotten_count} item(s) show rot/decay. Discard affected pieces.'
        batch_safe = False

    batch_summary = {
        'is_multi': is_multi,
        'total_items': total_items,
        'fresh_count': fresh_count,
        'rotten_count': rotten_count,
        'batch_safe': batch_safe,
        'batch_verdict': batch_verdict
    }

    return {
        'annotated_image': annotated_img,
        'items': item_results,
        'summary': batch_summary
    }
