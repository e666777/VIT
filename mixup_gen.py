import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# helper for mixup
def get_rand_bbox(width, height, l):
    r_x = np.random.randint(width)
    r_y = np.random.randint(height)
    r_l = np.sqrt(1 - l)
    r_w = np.int(width * r_l)
    r_h = np.int(height * r_l)
    return r_x, r_y, r_l, r_w, r_h

# custom image data generator
class MultiOutputDataGenerator(ImageDataGenerator):
    # custom image generator
    def __init__(self, featurewise_center = False, samplewise_center = False, 
                 featurewise_std_normalization = False, samplewise_std_normalization = False, 
                 zca_whitening = False, zca_epsilon = 1e-06, rotation_range = 0.0, width_shift_range = 0.0, 
                 height_shift_range = 0.0, brightness_range = None, shear_range = 0.0, zoom_range = 0.0, 
                 channel_shift_range = 0.0, fill_mode = 'nearest', cval = 0.0, horizontal_flip = False, 
                 vertical_flip = False, rescale = None, preprocessing_function = None, data_format = None, validation_split = 0.0, 
                 mix_up_alpha = 0.0, cutmix_alpha = 0.0): # additional class argument
    
        # parent's constructor
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization, samplewise_std_normalization, 
                         zca_whitening, zca_epsilon, rotation_range, width_shift_range, height_shift_range, brightness_range, 
                         shear_range, zoom_range, channel_shift_range, fill_mode, cval, horizontal_flip, vertical_flip, rescale, 
                         preprocessing_function, data_format, validation_split)

        # Mix-up
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        
        # Cutmix
        assert cutmix_alpha >= 0.0
        self.cutmix_alpha = cutmix_alpha

    def mix_up(self, X1, y1, X2, y2, ordered_outputs, target_lengths):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        batch_size = X1.shape[0]
        l = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        X_l = l.reshape(batch_size, 1, 1, 1)
        y_l = l.reshape(batch_size, 1)
        X = X1 * X_l + X2 * (1-X_l)
        target_dict = {}
        i = 0
        for output in ordered_outputs:
            target_length = target_lengths[output]
            target_dict[output] = y1[:, i: i + target_length] * y_l + y2[:, i: i + target_length] * (1 - y_l)
            i += target_length
        y = None
        for output, target in target_dict.items():
            if y is None:
                y = target
            else:
                y = np.concatenate((y, target), axis=1)
        return X, y
    
    def cutmix(self, X1, y1, X2, y2, ordered_outputs, target_lengths):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        width = X1.shape[1]
        height = X1.shape[0]
        r_x, r_y, r_l, r_w, r_h = get_rand_bbox(width, height, lam)
        bx1 = np.clip(r_x - r_w // 2, 0, width)
        by1 = np.clip(r_y - r_h // 2, 0, height)
        bx2 = np.clip(r_x + r_w // 2, 0, width)
        by2 = np.clip(r_y + r_h // 2, 0, height)
        X1[:, bx1:bx2, by1:by2, :] = X2[:, bx1:bx2, by1:by2, :]
        X = X1
        target_dict = {}
        i = 0
        for output in ordered_outputs:
            target_length = target_lengths[output]
            target_dict[output] = y1[:, i: i + target_length] * lam + y2[:, i: i + target_length] * (1 - lam)
            i += target_length
        y = None
        for output, target in target_dict.items():
            if y is None:
                y = target
            else:
                y = np.concatenate((y, target), axis=1)
        return X, y
    
    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        
        # for multi-outputs
        targets = None
        target_lengths = {}
        ordered_outputs = []
        for output, target in y.items():
            if targets is None:
                targets = target
            else:
                targets = np.concatenate((targets, target), axis=1)
            target_lengths[output] = target.shape[1]
            ordered_outputs.append(output)
        
        # parent flow
        batches = super().flow(x, targets, batch_size, shuffle, sample_weight, seed, save_to_dir, save_prefix, save_format, subset)
        
        # custom processing
        while True:
            batch_x, batch_y = next(batches)
            
            # mixup or cutmix
            if (self.mix_up_alpha > 0) & (self.cutmix_alpha > 0):
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                if np.random.rand() < 0.5:
                    batch_x, batch_y = self.mix_up(batch_x, batch_y, batch_x_2, batch_y_2, ordered_outputs, target_lengths)
                else:
                    batch_x, batch_y = self.cutmix(batch_x, batch_y, batch_x_2, batch_y_2, ordered_outputs, target_lengths)
            
                target_dict = {}
                i = 0
                for output in ordered_outputs:
                    target_length = target_lengths[output]
                    target_dict[output] = batch_y[:, i: i + target_length]
                    i += target_length
                    
                yield batch_x, target_dict
