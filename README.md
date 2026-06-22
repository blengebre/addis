
max_in  (encoder seq length): 153
max_out (decoder seq length): 74
Config saved to /home/genesistwo/newe/addis/config.json
Encoder input shape:  (117033, 153)
Decoder input shape:  (117033, 73)
Decoder target shape: (117033, 73)
E0000 00:00:1781701928.366541   47780 cuda_executor.cc:1737] INTERNAL: CUDA Runtime error: Failed call to cudaGetRuntimeVersion: Error loading CUDA libraries. GPU will not be used.: Error loading CUDA libraries. GPU will not be used.
W0000 00:00:1781701928.367153   47963 cuda_executor.cc:1755] Failed to determine cuDNN version (Note that this is expected if the application doesn't link the cuDNN plugin): INTERNAL: cuDNN error: CUDNN_STATUS_INTERNAL_ERROR
W0000 00:00:1781701928.389937   47780 gpu_device.cc:2365] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
Model: "functional"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                  ┃ Output Shape              ┃         Param # ┃ Connected to               ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ encoder_input (InputLayer)    │ (None, 153)               │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding (Embedding)         │ (None, 153, 128)          │         512,000 │ encoder_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add (Add)                     │ (None, 153, 128)          │               0 │ embedding[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention          │ (None, 153, 128)          │          66,048 │ add[0][0], add[0][0]       │
│ (MultiHeadAttention)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_1 (Dropout)           │ (None, 153, 128)          │               0 │ multi_head_attention[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_1 (Add)                   │ (None, 153, 128)          │               0 │ add[0][0], dropout_1[0][0] │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization           │ (None, 153, 128)          │             256 │ add_1[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense (Dense)                 │ (None, 153, 512)          │          66,048 │ layer_normalization[0][0]  │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_1 (Dense)               │ (None, 153, 128)          │          65,664 │ dense[0][0]                │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_2 (Dropout)           │ (None, 153, 128)          │               0 │ dense_1[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_2 (Add)                   │ (None, 153, 128)          │               0 │ layer_normalization[0][0], │
│                               │                           │                 │ dropout_2[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_1         │ (None, 153, 128)          │             256 │ add_2[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_1        │ (None, 153, 128)          │          66,048 │ layer_normalization_1[0][… │
│ (MultiHeadAttention)          │                           │                 │ layer_normalization_1[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_4 (Dropout)           │ (None, 153, 128)          │               0 │ multi_head_attention_1[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ decoder_input (InputLayer)    │ (None, 73)                │               0 │ -                          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_3 (Add)                   │ (None, 153, 128)          │               0 │ layer_normalization_1[0][… │
│                               │                           │                 │ dropout_4[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ embedding_1 (Embedding)       │ (None, 73, 128)           │         512,000 │ decoder_input[0][0]        │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_2         │ (None, 153, 128)          │             256 │ add_3[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_5 (Add)                   │ (None, 73, 128)           │               0 │ embedding_1[0][0]          │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_2 (Dense)               │ (None, 153, 512)          │          66,048 │ layer_normalization_2[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_2        │ (None, 73, 128)           │          66,048 │ add_5[0][0], add_5[0][0]   │
│ (MultiHeadAttention)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_3 (Dense)               │ (None, 153, 128)          │          65,664 │ dense_2[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_7 (Dropout)           │ (None, 73, 128)           │               0 │ multi_head_attention_2[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_5 (Dropout)           │ (None, 153, 128)          │               0 │ dense_3[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_6 (Add)                   │ (None, 73, 128)           │               0 │ add_5[0][0],               │
│                               │                           │                 │ dropout_7[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_4 (Add)                   │ (None, 153, 128)          │               0 │ layer_normalization_2[0][… │
│                               │                           │                 │ dropout_5[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_4         │ (None, 73, 128)           │             256 │ add_6[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_3         │ (None, 153, 128)          │             256 │ add_4[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_3        │ (None, 73, 128)           │          66,048 │ layer_normalization_4[0][… │
│ (MultiHeadAttention)          │                           │                 │ layer_normalization_3[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_9 (Dropout)           │ (None, 73, 128)           │               0 │ multi_head_attention_3[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_7 (Add)                   │ (None, 73, 128)           │               0 │ layer_normalization_4[0][… │
│                               │                           │                 │ dropout_9[0][0]            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_5         │ (None, 73, 128)           │             256 │ add_7[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_4 (Dense)               │ (None, 73, 512)           │          66,048 │ layer_normalization_5[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_5 (Dense)               │ (None, 73, 128)           │          65,664 │ dense_4[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_10 (Dropout)          │ (None, 73, 128)           │               0 │ dense_5[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_8 (Add)                   │ (None, 73, 128)           │               0 │ layer_normalization_5[0][… │
│                               │                           │                 │ dropout_10[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_6         │ (None, 73, 128)           │             256 │ add_8[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_4        │ (None, 73, 128)           │          66,048 │ layer_normalization_6[0][… │
│ (MultiHeadAttention)          │                           │                 │ layer_normalization_6[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_12 (Dropout)          │ (None, 73, 128)           │               0 │ multi_head_attention_4[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_9 (Add)                   │ (None, 73, 128)           │               0 │ layer_normalization_6[0][… │
│                               │                           │                 │ dropout_12[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_7         │ (None, 73, 128)           │             256 │ add_9[0][0]                │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ multi_head_attention_5        │ (None, 73, 128)           │          66,048 │ layer_normalization_7[0][… │
│ (MultiHeadAttention)          │                           │                 │ layer_normalization_3[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_14 (Dropout)          │ (None, 73, 128)           │               0 │ multi_head_attention_5[0]… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_10 (Add)                  │ (None, 73, 128)           │               0 │ layer_normalization_7[0][… │
│                               │                           │                 │ dropout_14[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_8         │ (None, 73, 128)           │             256 │ add_10[0][0]               │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_6 (Dense)               │ (None, 73, 512)           │          66,048 │ layer_normalization_8[0][… │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_7 (Dense)               │ (None, 73, 128)           │          65,664 │ dense_6[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dropout_15 (Dropout)          │ (None, 73, 128)           │               0 │ dense_7[0][0]              │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ add_11 (Add)                  │ (None, 73, 128)           │               0 │ layer_normalization_8[0][… │
│                               │                           │                 │ dropout_15[0][0]           │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ layer_normalization_9         │ (None, 73, 128)           │             256 │ add_11[0][0]               │
│ (LayerNormalization)          │                           │                 │                            │
├───────────────────────────────┼───────────────────────────┼─────────────────┼────────────────────────────┤
│ dense_8 (Dense)               │ (None, 73, 4000)          │         516,000 │ layer_normalization_9[0][… │
└───────────────────────────────┴───────────────────────────┴─────────────────┴────────────────────────────┘
 Total params: 2,465,696 (9.41 MB)
 Trainable params: 2,465,696 (9.41 MB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1097s 149ms/step - loss: 4.4860 - masked_accuracy: 0.3324 - val_loss: 3.3825 - val_masked_accuracy: 0.4539
Epoch 2/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1056s 144ms/step - loss: 3.0546 - masked_accuracy: 0.4993 - val_loss: 2.6290 - val_masked_accuracy: 0.5541
Epoch 3/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1055s 144ms/step - loss: 2.5501 - masked_accuracy: 0.5634 - val_loss: 2.3372 - val_masked_accuracy: 0.5896
Epoch 4/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1055s 144ms/step - loss: 2.3072 - masked_accuracy: 0.5941 - val_loss: 2.1899 - val_masked_accuracy: 0.6103
Epoch 5/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1055s 144ms/step - loss: 2.1655 - masked_accuracy: 0.6126 - val_loss: 2.0983 - val_masked_accuracy: 0.6235
Epoch 6/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1057s 144ms/step - loss: 2.0691 - masked_accuracy: 0.6264 - val_loss: 2.0404 - val_masked_accuracy: 0.6292
Epoch 7/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1059s 145ms/step - loss: 1.9930 - masked_accuracy: 0.6372 - val_loss: 1.9962 - val_masked_accuracy: 0.6386
Epoch 8/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1104s 145ms/step - loss: 1.9318 - masked_accuracy: 0.6462 - val_loss: 1.9688 - val_masked_accuracy: 0.6444
Epoch 9/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1067s 146ms/step - loss: 1.8829 - masked_accuracy: 0.6537 - val_loss: 1.9361 - val_masked_accuracy: 0.6480
Epoch 10/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1070s 146ms/step - loss: 1.8400 - masked_accuracy: 0.6600 - val_loss: 1.9083 - val_masked_accuracy: 0.6535
Epoch 11/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1073s 147ms/step - loss: 1.8010 - masked_accuracy: 0.6658 - val_loss: 1.8877 - val_masked_accuracy: 0.6555
Epoch 12/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1076s 147ms/step - loss: 1.7660 - masked_accuracy: 0.6712 - val_loss: 1.8742 - val_masked_accuracy: 0.6575
Epoch 13/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1083s 148ms/step - loss: 1.7372 - masked_accuracy: 0.6756 - val_loss: 1.8518 - val_masked_accuracy: 0.6625
Epoch 14/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1093s 149ms/step - loss: 1.7091 - masked_accuracy: 0.6799 - val_loss: 1.8519 - val_masked_accuracy: 0.6645
Epoch 15/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1092s 149ms/step - loss: 1.6832 - masked_accuracy: 0.6841 - val_loss: 1.8290 - val_masked_accuracy: 0.6680
Epoch 16/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1091s 149ms/step - loss: 1.6600 - masked_accuracy: 0.6877 - val_loss: 1.8175 - val_masked_accuracy: 0.6691
Epoch 17/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1091s 149ms/step - loss: 1.6375 - masked_accuracy: 0.6913 - val_loss: 1.8132 - val_masked_accuracy: 0.6701
Epoch 18/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1091s 149ms/step - loss: 1.6199 - masked_accuracy: 0.6942 - val_loss: 1.7917 - val_masked_accuracy: 0.6730
Epoch 19/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1091s 149ms/step - loss: 1.6027 - masked_accuracy: 0.6973 - val_loss: 1.8004 - val_masked_accuracy: 0.6733
Epoch 20/20
7315/7315 ━━━━━━━━━━━━━━━━━━━━ 1089s 149ms/step - loss: 1.5834 - masked_accuracy: 0.7005 - val_loss: 1.7905 - val_masked_accuracy: 0.6736

TensorBoard log directory: /home/genesistwo/newe/addis/logs/fit/20260617-131208
Model saved to /home/genesistwo/newe/addis/amh_omo_transformer.keras

============================================================
QUICK TRANSLATION TEST (on training samples)
============================================================

  AM: ሁላችንም መልስ እንፈልጋለን።
  REF: hundi keenya deebii barbaanna.
  PRED: hundi keenya deebii barbaanna.

  AM: ቶም አሁንም ምሳ እየበላ እንደሆነ አስባለሁ።
  REF: toom ammallee laaqana nyaachaa jiraa jedheen yaada.
  PRED: toom ammallee laaqana nyaachaa jira jedheen yaada.

  AM: ጸጥ ያሉ ፊልሞችን እጠላለሁ።
  REF: fiilmii callisaa nan jibba.
  PRED: fiilmii callisan nan jibba.

  AM: በእውነቱ ምንም ሀሳብ የለኝም።
  REF: dhugaa dubbachuuf yaada tokkollee hin qabu.
  PRED: dhuguma yaada tokkollee hin qabu.

  AM: በዚህ የጸደይ ወቅት እነሱን ለማየት በጉጉት እጠባበቃለሁ።
  REF: birraa kana isaan arguuf hawwii guddaan eeggadha.
  PRED: yeroo rifeensa sanaa arguuf fedhii guddaan eeggachaa jira.


Computing BLEU on test set (50 samples)...
BLEU score: 0.2158