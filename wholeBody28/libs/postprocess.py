import numpy as np


class Postprocess:
    def __init__(self):

        self.classes = {
            # 0: 'Body',
            # 1: 'Adult',
            # 2: 'Child',
            3: ('Body(M)', (0, 0, 255)),
            4: ('Body(F)', (255, 0, 255)),
            # 5: 'Body_with_Wheelchair',
            # 6: 'Body_with_Crutches',
            # 7: 'Head',
            8: ('Head(F)', (0, 255, 0)),
            9: ('Head(RF)', (0, 255, 0)),
            10: ('Head(R)', (0, 255, 0)),
            11: ('Head(RB)', (0, 255, 0)),
            12: ('Head(B)', (0, 255, 0)),
            13: ('Head(LB)', (0, 255, 0)),
            14: ('Head(L)', (0, 255, 0)),
            15: ('Head(LF)', (0, 255, 0)),
            16: ('', (120,120,120)),#Face
            17: ('', (20,120,240)),#Eye
            18: ('',(190,200,70)),#Nose
            19: ('', (250,20,120)),#Mouth
            20: ('', (200,255,170)),#Ear
            21: ('', (100,100,0)),#shoulder
            22: ('', (100,100,0)),#elbow-disrsek
            # 23: 'Hand',
            24: ('Hand(L)', (245,123,34)),
            25: ('Hand(R)', (245,123,34)),
            26: ('', (100,100,0)),#knee
            27: ('Foot', (23,45,190))
        }

    def postprocess(self, output: np.ndarray, score_threshold: int=0.35):
        # outputshape is [?,7](batchno, class_id, score, x1,y1,x2,y2)
        output = output[output[..., 2] >= score_threshold]
        output = output[np.isin(output[..., 1], list(self.classes.keys()))]
        
        results = []

        for row in output:
            bbox = row[3:].astype(np.int32)
            class_name = self.classes[row[1]][0]
            color = self.classes[row[1]][1]
            results.append([class_name, color, bbox])

        return results


        
    




    