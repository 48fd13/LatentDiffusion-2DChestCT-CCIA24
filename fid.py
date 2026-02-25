import numpy as np
from scipy import linalg
import torch
from torch.nn.functional import adaptive_avg_pool2d
from pytorch_fid.inception import InceptionV3

class FID():
    '''This class calculates the Frechet Distance and
    heavily borrows from the pytorch implementation
    https://github.com/mseitzer/pytorch-fid
    
    Dims:
        64:   first max pooling features
        192:  second max pooling features
        768:  pre-aux classifier features
        2048: final average pooling features (default)
    '''
    def __init__(self, source_samples, device) -> None:
        
        # source_samples.shape is (N, D, H, W) where N is the total nb of images and D is nb of channels
        # all images must be in the range [0, 1]
        
        self.dims = 2048
        self.device = device
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[self.dims]
        self.model = InceptionV3([block_idx]).to(device)
        self.model.eval()
        print(f'[FID] Computing activations from {len(source_samples)} samples')
        self.mu, self.sigma = self.__calculate_stats__(source_samples)

    @torch.no_grad()
    def __calculate_stats__(self, samples, batch_size=40):
        activations = np.empty((len(samples), self.dims))
        start_idx = 0

        for i in range(0, len(samples), batch_size):

            end = min(i+batch_size, len(samples))
            batch = samples[i:end].to(self.device)
            pred = self.model(batch)[0]

            # If model output is not scalar, apply global spatial average pooling.
            # This happens if you choose a dimensionality not equal 2048.
            if pred.size(2) != 1 or pred.size(3) != 1:
                pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

            pred = pred.squeeze(3).squeeze(2).cpu().numpy()
            activations[start_idx:start_idx + pred.shape[0]] = pred
            start_idx = start_idx + pred.shape[0]

        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    def calculate_FID(self, samples):
        mu, sigma = self.__calculate_stats__(samples)
        fid_value = self.calculate_frechet_distance(self.mu, self.sigma, mu, sigma)
        return fid_value

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)


        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert sigma1.shape == sigma2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return (diff.dot(diff) + np.trace(sigma1)
                + np.trace(sigma2) - 2 * tr_covmean)
