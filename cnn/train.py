import tensorflow as tf
from tensorflow.keras import backend as K
from . import models, data_utils

def run_training():
    data_dir = "data/train"
    epochs = 50
    batch_size = 32


    print("TensorFlow Version:", tf.__version__)

    print(f"\n--- Preparing Data for Model 1 (38x38) from {data_dir} ---")
    train_gen_38, val_gen_38, test_gen_38 = data_utils.create_generators(
        data_dir, target_size=(38, 38), batch_size=batch_size
    )

    print("\n--- Building & Training Model 1 ---")
    model1 = models.build_model_38()
    history1 = model1.fit(
        train_gen_38, 
        validation_data=val_gen_38, 
        epochs=epochs, 
        verbose=1
    )
    
    data_utils.plot_history(history1, title="Model 1 Accuracy (38x38)")
    
    if test_gen_38:
        print("\nEvaluating Model 1 on Test Set...")
        test_loss1, test_acc1 = model1.evaluate(test_gen_38, verbose=1)
        print(f"Model 1 Test Accuracy: {test_acc1:.4f}")
        show_random_predictions(model1, test_gen_38)

    K.clear_session()

    print(f"\n--- Preparing Data for Model 2 (64x64) from {data_dir} ---")
    train_gen_64, val_gen_64, test_gen_64 = data_utils.create_generators(
        data_dir, target_size=(64, 64), batch_size=batch_size
    )

    print("\n--- Building & Training Model 2 ---")
    model2 = models.build_model_64()
    history2 = model2.fit(
        train_gen_64, 
        validation_data=val_gen_64, 
        epochs=epochs, 
        verbose=1
    )

    data_utils.plot_history(history2, title="Model 2 Accuracy (64x64)")

    if test_gen_64:
        print("\nEvaluating Model 2 on Test Set...")
        test_loss2, test_acc2 = model2.evaluate(test_gen_64, verbose=1)
        print(f"Model 2 Test Accuracy: {test_acc2:.4f}")
        show_random_predictions(model2, test_gen_64)

if __name__ == "__main__":
    # Simply run the function, no arguments needed
    run_training()