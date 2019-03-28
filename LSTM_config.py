TIME_STEPS = 8  # 输入的时间步数
INPUT_SIZE = 1  # 每步有多少数据
OUTPUT_SIZE = INPUT_SIZE  # 输出的维度
EPOCHS = 10  # 迭代次数
drop_out = 0.3
BATCH_SIZE = 64

# 用于lstm.py
filepath = 'myModel/lstm' + '_timesteps' + str(TIME_STEPS) \
                          + '_input' + str(INPUT_SIZE) \
                          + '_epochs' + str(EPOCHS) \
                          + '_dropout' + str(drop_out) \
                          + '_batchsize' + str(BATCH_SIZE) \
                          + '_v1'
model_filepath = filepath +'.h5'
checkpoint_filepath = filepath + '_checkpoint'
tensorboard_filepath = filepath + '_log'

# 用于plot_lstm.py
load_model_file = 'myModel/lstm_timesteps8_input1_epochs10_dropout0.3_batchsize64_v1.h5'
