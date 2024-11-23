import numpy as np
from PIL import Image, ImageOps
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import copy
import json
import numpy as np
from typing import Tuple
import pickle
from collections import defaultdict
from collections import Counter

class Layout:

    def parse_table(self, original_table, cv_image_table, output_dir, idx):

        def isBoldText(image: cv2.Mat, contour: np.ndarray) -> bool:
            '''
                @brief check whether contour is black text or not
                @param contour: contour area, image 
                @return isBold: True or False
            '''
            x, y, w, h = cv2.boundingRect(contour)
            ct_img = image[y:y+h, x:x+w]

            num_white = np.sum(ct_img == 255)
            num_black = np.sum(ct_img == 0)

            if num_black > num_white:
                return True
            else:
                return False
        ##########################################

        def calRect(contours: np.ndarray) -> Tuple[int, int, int, int]:
            '''
                @brief calculate rectangle
                @param contours: contour area
                @return x, y, width, height
            '''
            x, y, w, h = cv2.boundingRect(contours)
            if __resizeFlag:
                w_diff = __orgWidth / __resizeRatio
                h_diff = __orgHeight / __resizeRatio
                x = int(x * w_diff)
                y = int(y * h_diff)
                w = int(w * w_diff)
                h = int(h * h_diff)
            return x, y, w, h

        ##########################################
        def arrange_cells(cells: dict):

            def unify_cells_by_mode(cells, key, threshold=3):

                groups = []
                
                # Group cells by proximity to the specified key
                for cell in cells:
                    added = False
                    for group in groups:
                        if abs(group[0][key] - cell[key]) <= threshold:
                            group.append(cell)
                            added = True
                            break
                    if not added:
                        groups.append([cell])

                # Unify values in each group using mode
                for group in groups:
                    # Find the mode of the key values
                    values = [cell[key] for cell in group]
                    mode_value = Counter(values).most_common(1)[0][0]
                    
                    # Update key values and add 'real_<key>' if modified
                    for cell in group:
                        if cell[key] != mode_value:
                            cell[f"real_{key}"] = cell[key]
                            cell[key] = mode_value

            def group_cells_by_key(cells, key):
                grouped = defaultdict(list)
                for cell in cells:
                    grouped[cell[key]].append(cell)
                return grouped

            def detect_merged_and_dimension_mapping(grouped_cells, size_key, dimension_key):
                merged_cells = []
                dimension_to_size = {}
                for dimension, cells in grouped_cells.items():
                    sizes = [cell[size_key] for cell in cells]
                    min_size = min(sizes)
                    for cell in cells:
                        if cell[size_key] > min_size * 1.5:  
                            merged_cells.append(cell)
                        else:
                            dimension_to_size[dimension] = cell[size_key]
                return merged_cells, dimension_to_size

            def create_next_key_mapping(sorted_keys):
                return {sorted_keys[i]: sorted_keys[i + 1] for i in range(len(sorted_keys) - 1)}

            def split_merged_cells(merged_cells, dimension_to_size, next_mapping, split_key, size_key, additional_key, real_coord):
                for merged_cell in merged_cells:
                    merged_size = merged_cell[size_key]
                    initial_size = dimension_to_size[merged_cell[split_key]]
                    next_point = next_mapping.get(merged_cell[split_key])

                    while merged_size - initial_size > 50 and next_point is not None:
                        cell = {
                            "x": merged_cell["x"],
                            "y": next_point,
                            "width": merged_cell["width"],
                            "height": merged_cell["height"]
                        }

                        if real_coord in merged_cell:
                            cell[additional_key] = merged_cell[real_coord]
                        else:
                            cell[additional_key] = merged_cell[split_key]
                        
                        initial_size += dimension_to_size[next_point]
                        cells.append(cell)
                        next_point = next_mapping.get(next_point)

            def process_cells(cells):
                
                unify_cells_by_mode(cells, 'y')
                unify_cells_by_mode(cells, 'x')

                
                y_grouped = group_cells_by_key(cells, "y")
                merged_cells_y, y_to_height = detect_merged_and_dimension_mapping(y_grouped, "height", "y")

                
                unique_y_values = sorted(y_to_height.keys())
                y_to_next_y = create_next_key_mapping(unique_y_values)

                
                split_merged_cells(merged_cells_y, y_to_height, y_to_next_y, "y", "height", "merged_cell_y", 'real_y')

                
                x_grouped = group_cells_by_key(cells, "x")
                merged_cells_x, x_to_width = detect_merged_and_dimension_mapping(x_grouped, "width", "x")

                
                unique_x_values = sorted(x_to_width.keys())
                x_to_next_x = create_next_key_mapping(unique_x_values)

                
                split_merged_cells(merged_cells_x, x_to_width, x_to_next_x, "x", "width", "merged_cell_x", 'real_x')

                cells_sorted = sorted(cells, key=lambda d: (d['y'], -d['x']))

                return cells_sorted

            cells = [box for box in cells if box['height'] > 40 and box['width'] > 80]
            processed_cells = process_cells(cells)
            y_counts = Counter(cell['y'] for cell in processed_cells)
            if y_counts:
                max_y = max(y_counts, key=y_counts.get)
                max_count = y_counts[max_y]
            else:
                max_count = 0
            
            return processed_cells, max_count
                     
        ##########################################
        
        def saveResults(exportImg: cv2.Mat, exportDict: dict, output_dir, idx) -> None:
            '''
                @brief save results
                @param exportImg: image to be saved
                    exportDict: json data to be saved
            '''

            output_dir = output_dir + '/' + 'cells_' + str(idx)
            os.makedirs(output_dir, exist_ok=True)

            cells_coords = exportDict["results"]
            cells_coords, n_cols = arrange_cells(cells_coords)

            for cell_number, box in enumerate(cells_coords):
                if 'merged_cell_y' in box:
                    y = box['merged_cell_y']
                elif 'real_y' in box:
                    y = box['real_y']
                else:
                    y = box['y']
                
                if 'merged_cell_x' in box:
                    x = box['merged_cell_x']
                elif 'real_x' in box:
                    x = box['real_x']
                else:
                    x = box['x']
                w, h = box['width'], box['height']
                xmin, ymin, xmax, ymax = (x, y, x + w, y + h)
                cell_image = exportImg.crop((xmin, ymin, xmax, ymax))
                cell_image.save(os.path.join(output_dir, f'cell_{cell_number}.png'))

            
            
            with open(os.path.join(output_dir, 'cells_coords.json'), "w") as json_file:
                json.dump(cells_coords, json_file, indent=4)
            
            
            with open(os.path.join(output_dir, 'n_cols.pkl'), 'wb') as file:
                # Write the variable to the pickle file
                pickle.dump(n_cols, file)

            print(f"Saved {len(cells_coords)} cell images in '{output_dir}' folder.")
            with open(os.path.join(output_dir, 'n_cells.pkl'), 'wb') as file:
                # Write the variable to the pickle file
                pickle.dump(len(cells_coords), file)
        ##########################################
        
        def contourDetection(processedImg: cv2.Mat) -> Tuple[cv2.Mat, dict]:
            '''
                @brief detect contours on image
                @param processedImg: empty table without text image
                @return contour drew image, contour area dict
            '''
            min_width = 5
            min_height = 5
            exportImg = copy.deepcopy(__image)

            exportDict = {
                'width': __orgWidth,
                'height': __orgHeight,
                'results': []
            }
            contours, hierarchy = cv2.findContours(
                processedImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for i in range(len(contours)):
                color = np.random.randint(0, 255, 3).tolist()

                if cv2.contourArea(contours[i]) < 200 or isBoldText(processedImg, contours[i]):
                    continue

                if hierarchy[0][i].any() != -1:
                    x, y, w, h = calRect(contours[i])
                    if w > min_width and h > min_height:
                        exportDict['results'].append(
                            {'x': x, 'y': y, 'width': w, 'height': h})
                        cv2.rectangle(exportImg, (x, y), (x+w, y+h), color, 2)

            # showImage(exportImg, 'exportImg')
            return exportImg, exportDict
        ##########################################

        def extractBoundary(image: cv2.Mat):
            '''
                @brief extract table boundaries by extracting verticle and horizontal lines
                @param image: cv2 read image
                @return empty table
            '''
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            invertImg = 255-gray
            _, edges = cv2.threshold(
                invertImg, 200, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

            vertical_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (1, np.array(image).shape[1]//100))
            vertical_lines = cv2.erode(edges, vertical_kernel, iterations=3)
            vertical_lines = cv2.dilate(
                vertical_lines, vertical_kernel, iterations=3)

            hor_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (np.array(image).shape[1]//100, 1))
            horizontal_lines = cv2.erode(edges, hor_kernel, iterations=3)
            horizontal_lines = cv2.dilate(
                horizontal_lines, hor_kernel, iterations=3)

            empty_table = cv2.addWeighted(
                vertical_lines, 0.5, horizontal_lines, 0.5, 0.0)
            empty_table = cv2.erode(~empty_table, kernel, iterations=2)
            _, empty_table = cv2.threshold(
                empty_table, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            #showImage(empty_table, 'table')
            return empty_table
        ##########################################
        def resizeImg(image: cv2.Mat) -> cv2.Mat:
            '''
                @brief resize image
                @param: cv2 read image, resize ratio
                @return: resized image, resize flag
            '''
            if __orgWidth < __resizeRatio and __orgHeight < __resizeRatio:
                image = cv2.resize(
                    image, (__resizeRatio, __resizeRatio), interpolation=cv2.INTER_CUBIC)
                __resizeFlag = True
            return image
        ##########################################

        def detectHeader(exportImg: cv2.Mat, exportDict: dict, output_dir, idx):
            cells_coords = exportDict["results"]
            cells_coords = [box for box in cells_coords if box['height'] > 40 and box['width'] > 80]
            xs = [coord['x'] + coord['width'] for coord in cells_coords]
            ys = [coord['y'] for coord in cells_coords]
            if len(ys) != 0:
                xmax = max(xs)
                ymin = min(ys)
                if ymin > 50:
                    output_dir = output_dir + '/' + 'cells_' + str(idx)
                    header = exportImg.crop((2, 0, xmax, ymin))
                    header.save(os.path.join(output_dir, 'header.png'))
                    print('Header Saved')
                else:
                    print('No Header')


        global __resizeRatio
        global __resizeFlag
        __resizeRatio = 640
        __resizeFlag = False
        global __image
        __image = cv_image_table
        global __orgHeight
        global __orgWidth
        __orgHeight, __orgWidth, _ = __image.shape
        __resize = resizeImg(__image)
        processedImg = extractBoundary(__resize)
        exportImg, exportDict = contourDetection(processedImg)
        saveResults(original_table, exportDict, output_dir, idx)
        detectHeader(original_table, exportDict, output_dir, idx)
##########################################

    def split_components(self, image_path, image_name, image, tables_coords, n_tables):

        output_dir = image_path + '/' + image_name
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'n_tables.pkl'), 'wb') as file:
            # Write the variable to the pickle file
            pickle.dump(n_tables, file)
        i = 0
        cv_image = np.array(image)
        cv_image = cv_image[:, :, ::-1].copy()

        sorted_coords_with_indices = sorted(enumerate(tables_coords), key=lambda x: x[1][1])
        sorted_indices = [index for index, _ in sorted_coords_with_indices]
        sorted_coords = [coords for _, coords in sorted_coords_with_indices]

        for idx, box in enumerate(sorted_coords):
            xmin, ymin, xmax, ymax = box
            cropped_image = Image.fromarray(cv_image[int(ymin):int(ymax), int(xmin):int(xmax)])

            cropped_image.save(os.path.join(output_dir, f'table_{idx}.png'))

            self.parse_table(cropped_image, cv2.cvtColor(np.array(cropped_image), cv2.COLOR_RGB2BGR), output_dir, idx)

            cv_image[int(ymin):int(ymax), int(xmin):int(xmax)] = 255
            midpoint = int((ymin + ymax) / 2)

            if idx == 0:
                i = i + 1
                piece = cv_image[:midpoint, :]
                piece_image = Image.fromarray(piece)
                
                piece_image.save(os.path.join(output_dir, f'part_{i}.png'))

            if idx != 0:
                i = i + 1
                piece = cv_image[previous_midpoint:midpoint, :]
                piece_image = Image.fromarray(piece)
                
                piece_image.save(os.path.join(output_dir, f'part_{i}.png'))
            previous_midpoint = midpoint

            if idx + 1 == n_tables:
                i = i + 1
                piece = cv_image[midpoint:, :]
                piece_image = Image.fromarray(piece)
                
                piece_image.save(os.path.join(output_dir, f'part_{i}.png'))

    def detect_table(self, image_path, image_name, orignal_image):

        gray = cv2.cvtColor(orignal_image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)

        # Detect table structure using morphological operations
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0)

        # Find contours (cells) to detect tables
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Set minimum dimensions for a valid table
        min_table_width = 500   # Minimum width to consider a contour as a table
        min_table_height = 95  # Minimum height to consider a contour as a table
        valid_contours = [cnt for cnt in contours if cv2.boundingRect(cnt)[2] >= min_table_width and cv2.boundingRect(cnt)[3] >= min_table_height]
        table_index = len(valid_contours)

        tables_coords = []
        if table_index:
            for cnt in valid_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                tables_coords.append((x, y, x + w, y + h))

        
            print(f"{table_index} tables detected in {image_path + '/' + image_name}.")
            self.split_components(image_path, image_name, orignal_image, tables_coords, table_index)

    def __call__(self,
                 image_path,
                 image_name,
                 ext
                ):
        
        full_image_path = fr"{image_path}/{image_name}{ext}"
        #print(full_image_path)
        image = cv2.imread(full_image_path)
        if image is None or image.size == 0:
            raise ValueError("Image is empty or not loaded correctly.")
        self.detect_table(image_path, image_name, image)
