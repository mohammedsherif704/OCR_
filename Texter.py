import easyocr
from PIL import Image, ImageOps
import numpy as np
import os
import pickle



class SeekText:

    def __init__(self):
        self.reader = easyocr.Reader(['ar', 'en'])

    def read_text(self, image_path, image_name, ext, text_file, new_line = False, cell = False):
        image_path = image_path + '/' + image_name + ext
        image = Image.open(image_path)

        preprocessed_image = image.convert('L')
        preprocessed_image = ImageOps.autocontrast(preprocessed_image, cutoff=0)

        preprocessed_image_np = np.array(preprocessed_image)

        result = self.reader.readtext(preprocessed_image_np, width_ths=200, slope_ths = 1, height_ths=10)

        ###################################
        sorted_result = sorted(result, key=lambda x: (x[0][0][1], x[0][0][0]))
        ###################################
        boxes_ys = [[i[1] for i in box[0]] for box in sorted_result ]
        boxes_max_xs = [max([i[0] for i in box[0]]) for box in sorted_result ]
        ###################################
        lines = [[(box[0] + box[2]) / 2, (box[0] + box[2]) / 2] for box in boxes_ys ]
        lines = sorted(list(set([item for sublist in lines for item in sublist])))
        ###################################
        all = []
        for i, line in enumerate(lines):
            li = []
            for j, box in enumerate(boxes_ys):
                if int(line) in range(int(np.ceil(box[0])), int(np.ceil(box[2] + 1))) or int(line) in range(int(np.ceil(box[1])), int(np.ceil(box[3] + 1))):
                    li.append(j)
            all.append(li)
        ###################################
        lll = []
        ccc = []
        for i in all:
            dic = set()
            for j in i:
                for idx, k in enumerate(all):
                    if idx not in lll and j in k:
                        dic.update(k)
                        lll.append(idx)
            if dic:
                ccc.append(list(dic))

        if not cell:
            with open(text_file, "a") as file:
                for l in ccc:
                    sorted_indexes = sorted(l, key=lambda i: boxes_max_xs[i], reverse=True)
                    concate = []
                    for i in sorted_indexes:
                        concate.append(sorted_result[i][1])
                    result = " ".join(concate)
                    file.write(result + "\n")
        else:
            all_lines = []
            for l in ccc:
                sorted_indexes = sorted(l, key=lambda i: boxes_max_xs[i], reverse=True)
                concate = []
                for i in sorted_indexes:
                    concate.append(sorted_result[i][1])
                result = " ".join(concate)
                all_lines.append(result)
            full_result = " ".join(all_lines)
            if new_line is True:
                with open(text_file, 'a') as file:
                     file.write('|    ' + full_result + '    |' + '\n')
                    
            else:
                with open(text_file, 'a') as file:
                    file.write('|    ' + full_result + '    |')

    def navigate_parts_folder(self, image_parts_path, text_file):

        def navigate_cells_folder(table_idx):
            cells_folder = image_parts_path + '/' + f'cells_{str(table_idx)}'
            with open(cells_folder + '/' + 'n_cols.pkl', 'rb') as file:
                 number_of_cols = pickle.load(file)
            with open(cells_folder + '/' + 'n_cells.pkl', 'rb') as file:
                 number_of_cells = pickle.load(file)
            cols_iter = 1
            for idx in range(number_of_cells):
                if cols_iter == number_of_cols:
                    self.read_text(cells_folder, f'cell_{idx}', '.png', text_file, new_line = True, cell = True)
                    cols_iter = 1
                else:
                    self.read_text(cells_folder, f'cell_{idx}', '.png', text_file, cell = True)
                    cols_iter = cols_iter + 1

        with open(image_parts_path + '/' + 'n_tables.pkl', 'rb') as file:
                 number_of_tables = pickle.load(file)
        parts_count = number_of_tables + 1

        for idx in range(1, parts_count):
            self.read_text(image_parts_path, f'part_{idx}', '.png', text_file)
            navigate_cells_folder(idx - 1)
        self.read_text(image_parts_path, f'part_{parts_count}', '.png', text_file)

    def __call__(self,
                 image_path,
                 image_name,
                 ext
                ):
        parts_folder = image_path + '/' + image_name
        text_file = image_path + '/' + image_name + '.txt'
        with open(text_file, 'w') as file:
            pass
        if os.path.exists(parts_folder) and os.path.isdir(parts_folder):
            self.navigate_parts_folder(parts_folder, text_file)
        else:
            self.read_text(image_path, image_name, '.png', text_file)
