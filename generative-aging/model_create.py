import torch
import numpy as np

import networks


class InferenceModel:
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    traverse = False  # This tells the model to traverse the latent space between anchor classes
    gpu_ids = [0]
    compare_to_trained_outputs = False
    deploy = False
    numClasses = 6  # number of age groups
    duplicate = 50
    cond_length = 300

    def initialize(self):
        torch.backends.cudnn.benchmark = True

        ##### define networks
        # Generators
        self.netG = networks.define_G(
            input_nc=3,
            output_nc=3,
            ngf=64,
            n_downsample_global=2,
            id_enc_norm='pixel',
            gpu_ids=self.gpu_ids,
            padding_type='reflect',
            style_dim=300,
            init_type='kaiming',
            conv_weight_norm=True,
            decoder_norm='pixel',
            activation='lrelu',
            adaptive_blocks=4,
            normalize_mlp=True,
            modulated_conv=True,
        )

        # load networks

        print("loading netG")
        self.netG.load_state_dict(
            torch.load('/mnt/checkpoints/latest_net_g_running.pth')
        )

    def set_inputs(self, data: dict):
        # TODO: Simplify the code some more
        # set input data to feed to the network
        inputs = data['Imgs']
        if inputs.dim() > 4:
            inputs = inputs.squeeze(0)

        self.class_A = data['Classes']
        if self.class_A.dim() > 1:
            self.class_A = self.class_A.squeeze(0)

        self.valid = torch.ones(1, dtype=torch.bool)

        self.image_paths = data['Paths']

        self.isEmpty = False if any(self.valid) else True
        if not self.isEmpty:
            available_idx = torch.arange(len(self.class_A))
            select_idx = torch.masked_select(available_idx, self.valid).long()
            inputs = torch.index_select(inputs, 0, select_idx)

            self.class_A = torch.index_select(self.class_A, 0, select_idx)
            self.image_paths = [
                val for i, val in enumerate(self.image_paths) if self.valid[i] == 1
            ]

        self.reals = inputs
        self.reals = self.reals.cuda()

    def get_conditions(self, mode: str = 'train'):
        # set conditional inputs to the network
        nb = self.numValid

        # tex condition mapping
        condG_A_gen = self.Tensor(nb, self.cond_length)
        condG_B_gen = self.Tensor(nb, self.cond_length)
        condG_A_orig = self.Tensor(nb, self.cond_length)
        condG_B_orig = self.Tensor(nb, self.cond_length)

        noise_sigma = 0.2

        for i in range(nb):
            condG_A_gen[i, :] = (noise_sigma * torch.randn(1, self.cond_length)).cuda()
            condG_A_gen[
                i,
                self.class_B[i]
                * self.duplicate : (self.class_B[i] + 1)
                * self.duplicate,
            ] += 1
            if not (self.traverse or self.deploy):
                condG_B_gen[i, :] = (
                    noise_sigma * torch.randn(1, self.cond_length)
                ).cuda()
                condG_B_gen[
                    i,
                    self.class_A[i]
                    * self.duplicate : (self.class_A[i] + 1)
                    * self.duplicate,
                ] += 1

                condG_A_orig[i, :] = (
                    noise_sigma * torch.randn(1, self.cond_length)
                ).cuda()
                condG_A_orig[
                    i,
                    self.class_A[i]
                    * self.duplicate : (self.class_A[i] + 1)
                    * self.duplicate,
                ] += 1

                condG_B_orig[i, :] = (
                    noise_sigma * torch.randn(1, self.cond_length)
                ).cuda()
                condG_B_orig[
                    i,
                    self.class_B[i]
                    * self.duplicate : (self.class_B[i] + 1)
                    * self.duplicate,
                ] += 1

        self.gen_conditions = condG_A_gen  # self.class_B

    def get_age_group(self, age: int) -> int:
        """
        Returns the index of the age group bucket to which the given age belongs.
        If the age does not belong to any bucket, it returns the index of the bucket closest to the age.
        Example:
        >>> get_age_group(1)
        0
        >>> get_age_group(22)
        3
        """
        buckets = [[0, 2], [3, 6], [7, 9], [15, 19], [30, 39], [50, 69]]

        # Check if age is in any bucket
        for i, bucket in enumerate(buckets):
            if bucket[0] <= age <= bucket[1]:
                return i

        # If age is not in any bucket, find the closest bucket
        closest_bucket_index = min(
            range(len(buckets)),
            key=lambda index: min(
                abs(age - buckets[index][0]), abs(age - buckets[index][1])
            ),
        )

        return closest_bucket_index

    @torch.inference_mode()
    def inference(self, data: dict, age: int = 80) -> np.ndarray:
        self.set_inputs(data)
        if self.isEmpty:
            return

        self.numValid = self.valid.sum().item()
        sz = self.reals.size()
        self.fake_B = self.Tensor(self.numClasses, sz[0], sz[1], sz[2], sz[3])

        idx = self.get_age_group(age)

        with torch.no_grad():
            self.class_B = self.Tensor(self.numValid).long().fill_(idx)
            self.get_conditions(mode='test')

            self.fake_B[idx, :, :, :, :] = self.netG.infer(
                self.reals, self.gen_conditions
            )

            output = tensor2im(self.fake_B[idx])

        return output

    def forward(self, data: dict) -> dict:
        return self.inference(data)


def tensor2im(image_tensor: torch.Tensor, imtype=np.uint8) -> np.ndarray:
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0

    return image_numpy.astype(imtype)


def create_model():
    model = InferenceModel()
    model.initialize()

    return model
