import cv2 as cv

def resize_for_display(image, width: int, height: int):
    if image is None:
        return image
    
    if width is None or height is None:
        return image
    
    if width <= 0 or height <= 0:
        return image
    
    ih, iw = image.shape[:2]
    if iw == width and ih == height:
        return image
    
    if width < iw or height < ih:
        interp = cv.INTER_AREA
    else:
        interp = cv.INTER_LINEAR
        
    try:
        return cv.resize(image, (width, height), interpolation=interp)
    except Exception:
        return image
