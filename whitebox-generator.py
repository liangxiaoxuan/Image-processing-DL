mport sys
import os
module_path = os.path.abspath(os.getcwd())
if module_path not in sys.path:
    sys.path.append(module_path)
sys.path.append('/tensorflow/')


"""
model: should use tf-keras
data: cifar 10
"""
import tensorflow as tf
from exp.keras_wraper_ensemble import KerasModelWrapper
from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod, \
    BasicIterativeMethod, DeepFool, ElasticNetMethod, FastFeatureAdversaries,\
    HopSkipJumpAttack, LBFGS, MadryEtAl, MaxConfidence, MomentumIterativeMethod, ProjectedGradientDescent, \
    SparseL1Descent, SPSA, SpatialTransformationMethod, Noise
from datasets.imagenet import ImagenetLoader
from tqdm import tqdm

# from cleverhans.utils_keras import KerasModelWrapper


def dataloader_Imagenet(batch_size):
    max_num_images = None
    max_num_images_per_class = 5
    dataLoader = ImagenetLoader(split='val', batch_size=batch_size, max_num_images=max_num_images,
                                max_num_images_per_class=max_num_images_per_class, shuffle=False)

    dataset_it = iter(dataLoader)
    nBatches = len(dataLoader)

    # placeholder = model_gtc.get_placeholder()

    for data_j in tqdm(dataset_it, total=nBatches):
        image, labels = data_j
        images = image

    return images, labels
  
  
  
  def whitebox_generator(model, sess, data_size, attack_way, placeholder, attack_type, attack_params):

    # model(model.input)
    wrap = KerasModelWrapper(model)
    #wrap = model

    clip_min = None
    clip_max = None
    # input_shape = x_test.shape[1:]
    # model_input = Input(shape=input_shape)

    eps_ = 0.1
    initial_const = 0.001
    overshoot = 0.001
    fgsm_params = {'eps': eps_,
                   'clip_min': clip_min,
                   'clip_max': clip_max}
    noise_params = {'eps': eps_,
                   'clip_min': clip_min,
                   'clip_max': clip_max}
    cw_params = {
        'batch_size': 20,
        'confidence': 0.1,
        'learning_rate': 0.01,
        'binary_search_steps': 1,
        'max_iterations': 1000,
        'initial_const': initial_const,
        "clip_min": -124,
        "clip_max": 155
        }
    
    # ATTTACK
    ATTACK_TYPE, ATTACK_PARAMS, = str(attack_type), str(attack_params)
    
    attacks_param_dict = {'fgsm': fgsm_params, "bim": bim_params, "deepfool": deepfool_params, "ead": ead_params,
                          "ffa": ffa_params,
                          "hsj": hsj_params, "lbfgs": lbfgs_params, "mad": mad_params, "maxconfi": maxconfi_params,
                          "mim": mim_params,
                          "noise": noise_params, "pgd": pgd_params,
                          "sd": sd_params,
                          "stm": stm_params, "spsa": spsa_params, "cw": cw_params}
    attacks_type_dict = {"FastGradientMethod": FastGradientMethod, "BasicIterativeMethod": BasicIterativeMethod,
                         "DeepFool": DeepFool, "ElasticNetMethod": ElasticNetMethod,
                         "FastFeatureAdversaries": FastFeatureAdversaries,
                         "HopSkipJumpAttack": HopSkipJumpAttack, "LBFGS": LBFGS, "MadryEtAl": MadryEtAl,
                         "MaxConfidence": MaxConfidence,
                         "MomentumIterativeMethod": MomentumIterativeMethod, "Noise": Noise,
                         "ProjectedGradientDescent": ProjectedGradientDescent,
                          "SparseL1Descent": SparseL1Descent,
                         "SpatialTransformationMethod": SpatialTransformationMethod, "SPSA": SPSA,
                          "CarliniWagnerL2": CarliniWagnerL2}

    if attack_way == "iterative":
        attacks = attacks_type_dict[ATTACK_TYPE](wrap, sess=sess)
        adv_x = tf.stop_gradient(attacks.generate(placeholder, **attacks_param_dict[ATTACK_PARAMS]))
        #adv_x = attacks.generate(model_input, **attacks_param_dict[ATTACK_PARAMS])

    elif attack_way == "optimazation":
        attacks = attacks_type_dict[ATTACK_TYPE](wrap, sess=sess)
        adv_x = attacks.generate(placeholder, **attacks_param_dict[ATTACK_PARAMS])
        
    return adv_x

  
if __name__ == '__main__'
      whitebox_generator()
