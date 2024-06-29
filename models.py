import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modality_Encoder import *

class MDNet(nn.Module):
    def __init__(self, config):
        super(MDNet, self).__init__()

        self.config = config
        self.EEGFeatureExtractor = EEG_Encoder(batch_size=config.batch_size, seq_len=3*128, n_channels=32, n_classes=2)
        self.EOGFeatureExtractor = EOG_Encoder(batch_size=config.batch_size, seq_len=3*128, n_channels=2, n_classes=2)
        self.EMGFeatureExtractor = EOG_Encoder(batch_size=config.batch_size, seq_len=3*128, n_channels=2, n_classes=2)

        self.eeg_input_size = 32
        self.eog_input_size = 2
        self.emg_input_size = 2

        self.eeg_hidden_size = config.hidden_size
        self.eog_hidden_size = config.hidden_size
        self.emg_hidden_size = config.hidden_size
        self.subject_num = subject_num = config.subject_num
        self.hidden_size = hidden_size = config.hidden_size

        self.input_sizes = input_sizes = [self.eeg_input_size, self.eog_input_size, self.emg_input_size]
        self.hidden_sizes = hidden_sizes = [int(self.eeg_hidden_size), int(self.eog_hidden_size), int(self.emg_hidden_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.ELU = nn.ELU()

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        self.project_eeg = nn.Sequential()
        self.project_eeg.add_module('project_eeg', nn.Linear(in_features=8 * 2 * 8,
                                                             out_features=config.hidden_size))
        self.project_eeg.add_module('project_eeg_activation', self.activation)
        self.project_eeg.add_module('project_eeg_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_eog = nn.Sequential()
        self.project_eog.add_module('project_eog', nn.Linear(in_features=8 * 2 * 8,
                                                             out_features=config.hidden_size))
        self.project_eog.add_module('project_eog_activation', self.activation)
        self.project_eog.add_module('project_eog_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_emg = nn.Sequential()
        self.project_emg.add_module('project_emg', nn.Linear(in_features=8 * 2 * 8,
                                                             out_features=config.hidden_size))
        self.project_emg.add_module('project_emg_activation', self.activation)
        self.project_emg.add_module('project_emg_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # Modality-specific Encoders
        ##########################################
        self.private_eeg = nn.Sequential()
        self.private_eeg.add_module('private_eeg_1',
                                    nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_eeg.add_module('private_eeg_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.private_eeg.add_module('private_eeg_activation_1', self.ELU)

        self.private_eog = nn.Sequential()
        self.private_eog.add_module('private_eog_1',
                                    nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_eog.add_module('private_eog_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.private_eog.add_module('private_eog_activation_1', self.ELU)

        self.private_emg = nn.Sequential()
        self.private_emg.add_module('private_emg_1',
                                    nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_emg.add_module('private_emg_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.private_emg.add_module('private_emg_activation_1', self.ELU)

        ##########################################
        # Modality-invariant Encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_batch_norm_1', nn.BatchNorm1d(config.hidden_size))
        self.shared.add_module('shared_activation_1', self.ELU)

        ##########################################
        # Modality reconstruct
        ##########################################
        self.recon_eeg = nn.Sequential()
        self.recon_eeg.add_module('recon_eeg_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        self.recon_eog = nn.Sequential()
        self.recon_eog.add_module('recon_eog_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        self.recon_emg = nn.Sequential()
        self.recon_emg.add_module('recon_emg_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # subject private encoders
        ##########################################
        self.subject_private = []
        for i in range(subject_num):
                self.subject_private.append(nn.Sequential(
                    nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6),
                    self.ELU,
                    nn.Dropout(p=self.dropout_rate),
                    nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6)
                ))
        ##########################################
        # subject shared encoder
        ##########################################
        self.subject_shared = nn.Sequential()
        self.subject_shared.add_module('shared_1', nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6))
        self.subject_shared.add_module('shared_activation_1', self.ELU)
        self.subject_shared.add_module('Dropout', nn.Dropout(p=self.dropout_rate))
        self.subject_shared.add_module('shared_2', nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6))
        self.subject_shared.add_module('shared_batch_norm_2', nn.BatchNorm1d(hidden_size * 6))

        ##########################################
        # subject reconstruct
        ##########################################
        self.recon_subject = nn.Sequential()
        self.recon_subject.add_module('recon', nn.Linear(in_features=hidden_size * 6, out_features=hidden_size * 6))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_batch_norm', nn.BatchNorm1d(self.hidden_size * 6))
        self.fusion.add_module('fusion_layer_1',
                               nn.Linear(in_features=self.hidden_size * 6, out_features=self.hidden_size))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(self.dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', nn.ELU())
        self.fusion.add_module('fusion_layer_3', nn.Linear(in_features=self.hidden_size, out_features=output_size))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, dim_feedforward=hidden_size*2,nhead=2,dropout=self.dropout_rate)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=1)

        self.utt_feature_norm = nn.BatchNorm1d(6 * self.hidden_size)


    def extract_features(self, extractor, x):
        features = extractor(x)
        return features


    def alignment(self, eeg, eog, emg, lengths, subject_labels):
        batch_size = lengths.size(0)

        self.eeg = eeg
        self.eog = eog
        self.emg = emg

        # extract features from eeg modality
        utterance_eeg = self.extract_features(self.EEGFeatureExtractor, eeg)
        # print(utterance_eeg.shape)
        # extract features from eog modality
        utterance_eog = self.extract_features(self.EOGFeatureExtractor, eog)
        # extract features from emg modality
        utterance_emg = self.extract_features(self.EMGFeatureExtractor, emg)
        # Shared-private encoders
        self.shared_private(utterance_eeg, utterance_eog, utterance_emg)

        ###############
        # For reconstruction
        self.reconstruct()

        # # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_eeg, self.utt_private_eog, self.utt_private_emg, self.utt_shared_eeg, self.utt_shared_eog,  self.utt_shared_emg), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0], h[1], h[2], h[3], h[4], h[5]), dim=1)
        h = self.utt_feature_norm(h)
        self.utt_subject = h
        self.utt_shared_subject = self.subject_shared(h)
        h = h.reshape(batch_size, 1, -1)
        self.utt_private_subject = torch.ones_like(h)

        for i in range(batch_size):
            self.utt_private_subject[i] = self.subject_private[subject_labels[i]](h[i])

        self.utt_private_subject = self.utt_private_subject.reshape(batch_size, -1)
        self.utt_subject_recon = self.recon_subject(self.utt_subject)
        h=h.reshape(self.utt_shared_subject.shape[0],self.utt_shared_subject.shape[1])
        o = self.fusion(self.utt_shared_subject+h)

        return o


    def reconstruct(self,):

        self.utt_eeg = (self.utt_private_eeg + self.utt_shared_eeg)
        self.utt_eog = (self.utt_private_eog + self.utt_shared_eog)
        self.utt_emg = (self.utt_private_emg + self.utt_shared_emg)

        self.utt_eeg_recon = self.recon_eeg(self.utt_eeg)
        self.utt_eog_recon = self.recon_eog(self.utt_eog)
        self.utt_emg_recon = self.recon_emg(self.utt_emg)


    def shared_private(self, utterance_eeg, utterance_eog, utterance_emg):

        # Projecting to same sized space
        self.utt_eeg_orig = utterance_eeg = self.project_eeg(utterance_eeg)
        self.utt_eog_orig = utterance_eog = self.project_eog(utterance_eog)
        self.utt_emg_orig = utterance_emg = self.project_emg(utterance_emg)

        # Private-shared components
        self.utt_private_eeg = self.private_eeg(utterance_eeg)
        self.utt_private_eog = self.private_eog(utterance_eog)
        self.utt_private_emg = self.private_emg(utterance_emg)

        self.utt_shared_eeg = self.shared(utterance_eeg)
        self.utt_shared_eog = self.shared(utterance_eog)
        self.utt_shared_emg = self.shared(utterance_emg)


    def forward(self, eeg, eog, emg, lengths, subject_labels):
        output = self.alignment(eeg, eog, emg, lengths, subject_labels)
        return output
