import cv2
import numpy as np
import pandas as pd
import imutils
import argparse

def get_answer_key(file_path):
    """
    Parses the provided CSV file to create a dictionary-based answer key.
    """
    answers = {}
    try:
        df = pd.read_csv(file_path)
        for col in df.columns:
            # Handle potential inconsistencies like '1 - a' or '1. a'
            if ' - ' in str(df[col][0]):
                for item in df[col]:
                    if isinstance(item, str):
                        try:
                            q_num, ans = item.split(' - ')
                            q_num = int(''.join(filter(str.isdigit, q_num)))
                            answers[q_num] = ans.strip().lower()
                        except ValueError:
                            continue
            elif '. ' in str(df[col][0]):
                for item in df[col]:
                    if isinstance(item, str):
                        try:
                            q_num, ans = item.split('. ')
                            q_num = int(''.join(filter(str.isdigit, q_num)))
                            answers[q_num] = ans.strip().lower()
                        except ValueError:
                            continue
    except FileNotFoundError:
        print(f"Error: The answer key file '{file_path}' was not found.")
        return None
    return answers

def main(image_path, key_path):
    """
    Main function to process the OMR sheet and score it.
    """
    answer_key = get_answer_key(key_path)
    if not answer_key:
        return

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read the image file '{image_path}'.")
        return

    # Resize image for faster processing
    image = imutils.resize(image, height=800)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use adaptive thresholding to handle uneven lighting
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find contours
    contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Filter contours to find potential bubbles
    bubble_contours = []
    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        # Filter based on bubble size and aspect ratio
        if w >= 20 and h >= 20 and ar >= 0.8 and ar <= 1.2:
            bubble_contours.append(c)

    # Sort contours from top-to-bottom
    bubble_contours = sorted(bubble_contours, key=lambda c: cv2.boundingRect(c)[1])

    # Group bubbles into rows (questions) and then columns
    questions = []
    current_row = []
    if bubble_contours:
        current_y = cv2.boundingRect(bubble_contours[0])[1]
        for c in bubble_contours:
            x, y, w, h = cv2.boundingRect(c)
            # Group bubbles that are vertically close to each other
            if abs(y - current_y) < 20: # A small threshold for row grouping
                current_row.append(c)
            else:
                # Sort the current row by x-coordinate to get the correct order of options (a, b, c, d)
                current_row.sort(key=lambda c: cv2.boundingRect(c)[0])
                questions.append(current_row)
                current_row = [c]
            current_y = y
        # Add the last row
        if current_row:
            current_row.sort(key=lambda c: cv2.boundingRect(c)[0])
            questions.append(current_row)

    # Now, process each question
    correct_answers = 0
    total_questions = len(questions)
    for q_num, question_contours in enumerate(questions, start=1):
        if q_num not in answer_key:
            continue

        filled_bubbles = []
        for i, c in enumerate(question_contours):
            mask = np.zeros(thresh.shape, dtype="uint8")
            cv2.drawContours(mask, [c], -1, 255, -1)
            mask = cv2.bitwise_and(thresh, thresh, mask=mask)
            total = cv2.countNonZero(mask)

            # A bubble is considered filled if the number of non-zero pixels is above a threshold
            # The threshold is 40% of the total bubble area.
            if total > (0.4 * cv2.contourArea(c)):
                filled_bubbles.append(chr(ord('a') + i))

        # Scoring Logic
        if len(filled_bubbles) == 1 and filled_bubbles[0] == answer_key[q_num]:
            correct_answers += 1
            color = (0, 255, 0)  # Green for correct
        else:
            color = (0, 0, 255)  # Red for incorrect

        # Draw a circle around the detected filled bubble
        if filled_bubbles:
            # Find the contour that was filled
            for i, c in enumerate(question_contours):
                if chr(ord('a') + i) in filled_bubbles:
                    (x, y, w, h) = cv2.boundingRect(c)
                    cv2.circle(image, (int(x + w / 2), int(y + h / 2)), 15, color, 2)

    print(f"Score: {correct_answers} / {total_questions}")
    cv2.imshow("Result", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automated OMR Evaluation and Scoring System")
    parser.add_argument("-i", "--image", required=True, help="Path to the input OMR sheet image")
    parser.add_argument("-k", "--key", required=True, help="Path to the answer key CSV file")
    args = vars(parser.parse_args())

    main(args["image"], args["key"])