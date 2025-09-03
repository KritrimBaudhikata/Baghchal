from keras._tf_keras.keras.layers import Input, Conv2D, Flatten, Dense, BatchNormalization, Dropout
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.layers import Multiply, Activation, Add, Softmax
from keras._tf_keras.keras.regularizers import l2
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import ReduceLROnPlateau, EarlyStopping


def create_bagh_chal_model(input_shape=(5, 5, 5), action_space_size=65, learning_rate=0.001):
    """
    Creates an enhanced neural network model for Bagh Chal.
    :param input_shape: The shape of the input state tensor.
    :param action_space_size: The size of the action space (number of possible moves).
    :param learning_rate: Initial learning rate for the optimizer.
    :return: A compiled Keras model with policy and value heads.
    
    Improvements:
    1. Enhanced architecture with more residual blocks
    2. Dropout layers for regularization
    3. Separate optimizers for policy and value heads
    4. Learning rate scheduling
    5. Better initialization and regularization
    """
    # Input Layer
    state_input = Input(shape=input_shape, name='state_input')
    mask_input = Input(shape=(action_space_size,), name='mask_input')

    # Enhanced Convolutional Layers with more filters
    conv = Conv2D(512, (3, 3), padding='same', activation='relu',
                  kernel_regularizer=l2(0.0001), name='conv_initial')(state_input)
    conv = BatchNormalization()(conv)
    conv = Dropout(0.1)(conv)

    # Enhanced Residual Blocks (increased from 5 to 8)
    for i in range(8):
        shortcut = conv
        conv = Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=l2(0.0001), name=f'conv_res_{i}')(conv)
        conv = BatchNormalization()(conv)
        conv = Dropout(0.1)(conv)
        conv = Conv2D(512, (3, 3), padding='same',
                      kernel_regularizer=l2(0.0001), name=f'conv_res_{i}_2')(conv)
        conv = BatchNormalization()(conv)
        conv = Add()([shortcut, conv])
        conv = Activation('relu')(conv)

    # Policy Head with enhanced architecture
    policy_conv = Conv2D(256, (1, 1), kernel_regularizer=l2(0.0001), name='policy_conv')(conv)
    policy_conv = BatchNormalization()(policy_conv)
    policy_conv = Activation('relu')(policy_conv)
    policy_conv = Dropout(0.1)(policy_conv)
    
    policy_flat = Flatten()(policy_conv)
    policy_hidden = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(policy_flat)
    policy_hidden = Dropout(0.1)(policy_hidden)
    policy_logits = Dense(action_space_size, activation='linear', kernel_regularizer=l2(0.0001))(policy_hidden)
    
    # Apply action mask
    masked_logits = Multiply()([policy_logits, mask_input])
    policy_output = Activation('softmax', name='policy')(masked_logits)

    # Value Head with enhanced architecture
    value_conv = Conv2D(256, (1, 1), kernel_regularizer=l2(0.0001), name='value_conv')(conv)
    value_conv = BatchNormalization()(value_conv)
    value_conv = Activation('relu')(value_conv)
    value_conv = Dropout(0.1)(value_conv)
    
    value_flat = Flatten()(value_conv)
    value_hidden1 = Dense(512, activation='relu', kernel_regularizer=l2(0.0001))(value_flat)
    value_hidden1 = Dropout(0.1)(value_hidden1)
    value_hidden2 = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(value_hidden1)
    value_hidden2 = Dropout(0.1)(value_hidden2)
    value_output = Dense(1, activation='tanh', name='value')(value_hidden2)

    # Create model
    model = Model(inputs=[state_input, mask_input], outputs=[policy_output, value_output])
    
    # Separate optimizers for better training
    policy_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    value_optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    
    # Compile with separate optimizers and loss weights
    model.compile(
        optimizer={'policy': policy_optimizer, 'value': value_optimizer},
        loss={'policy': 'categorical_crossentropy', 'value': 'mean_squared_error'},
        loss_weights={'policy': 1.0, 'value': 1.0},
        metrics={'policy': 'accuracy', 'value': 'mae'}
    )

    return model


def get_training_callbacks(patience=10, factor=0.5, min_lr=1e-7):
    """
    Get training callbacks for learning rate scheduling and early stopping.
    
    :param patience: Number of epochs to wait before reducing learning rate.
    :param factor: Factor by which to reduce learning rate.
    :param min_lr: Minimum learning rate.
    :return: List of callbacks.
    """
    callbacks = [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=patience * 2,
            restore_best_weights=True,
            verbose=1
        )
    ]
    return callbacks
