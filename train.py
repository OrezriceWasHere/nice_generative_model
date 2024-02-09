"""Training procedure for NICE.
"""

import argparse
import torch
import torchvision
import tqdm
from torchvision import transforms
import matplotlib.pyplot as plt
import nice


def train(flow, trainloader, optimizer, epoch):
    flow.train()  # set to training mode
    neg_log_likelihood = 0
    for inputs, _ in trainloader:
        inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[
            3])  # change  shape from BxCxHxW to Bx(C*H*W)

        inputs = inputs.to(flow.device)
        # TODO Fill in
        nll = flow.log_prob(inputs)
        avg_nll = -nll.mean()
        loss = avg_nll
        # loss = torch.abs(avg_nll)
        neg_log_likelihood += loss.item()
        loss.backward()
        optimizer.step()
        flow.zero_grad()

    return neg_log_likelihood / len(trainloader)


def test(flow, testloader, filename, epoch, sample_shape):
    flow.eval()  # set to inference mode
    neg_log_likelihood = 0
    with torch.no_grad():
        samples = flow.sample(100).cpu()
        a, b = samples.min(), samples.max()
        samples = (samples - a) / (b - a + 1e-10)
        samples = samples.view(-1, sample_shape[0], sample_shape[1], sample_shape[2])
        torchvision.utils.save_image(torchvision.utils.make_grid(samples),
                                     './samples/' + filename + 'epoch%d.png' % epoch)
        # TODO full in
        for inputs, _ in testloader:
            inputs = inputs.view(inputs.shape[0], inputs.shape[1] * inputs.shape[2] * inputs.shape[
                3])
            inputs = inputs.to(flow.device)
            log_likelihood = flow.log_prob(inputs)
            loss = -log_likelihood.mean()
            neg_log_likelihood += loss.item()

    test_loss = neg_log_likelihood / len(testloader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))

    return test_loss


def add_noise(x):
    return x + torch.zeros_like(x).uniform_(0., 1. / 256.)


def plot_draws(train_log_likelihood, test_log_likelihood):
    """This function draws matplotlib of train log likelihood and test"""
    # Using Numpy to create an array X
    X = list(range(len(train_log_likelihood)))

    # Assign variables to the y axis part of the curve


    # Plotting both the curves simultaneously
    plt.plot(X, train_log_likelihood, color='r', label='train')
    plt.plot(X, test_log_likelihood, color='g', label='test')

    # Naming the x-axis, y-axis and the whole graph
    plt.xlabel("epoch")
    plt.ylabel("average negative log likelihood per epoch")
    plt.title("training process")

    # Adding legend, which helps us recognize the curve according to it's color
    plt.legend()

    # To load the display window
    plt.show()


def main(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    sample_shape = [1, 28, 28]

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (1.,)),
        transforms.Lambda(add_noise)  # dequantization
    ])

    if args.dataset == 'mnist':
        trainset = torchvision.datasets.MNIST(root='./data/MNIST',
                                              train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.MNIST(root='./data/MNIST',
                                             train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    elif args.dataset == 'fashion-mnist':
        trainset = torchvision.datasets.FashionMNIST(root='~/torch/data/FashionMNIST',
                                                     train=True, download=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(trainset,
                                                  batch_size=args.batch_size, shuffle=True, num_workers=2)
        testset = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',
                                                    train=False, download=True, transform=transform)
        testloader = torch.utils.data.DataLoader(testset,
                                                 batch_size=args.batch_size, shuffle=False, num_workers=2)
    else:
        raise ValueError('Dataset not implemented')

    model_save_filename = '%s_' % args.dataset \
                          + 'batch%d_' % args.batch_size \
                          + 'coupling%d_' % args.coupling \
                          + 'coupling_type%s_' % args.coupling_type \
                          + 'mid%d_' % args.mid_dim \
                          + 'hidden%d_' % args.hidden \
                          + '.pt'

    flow = nice.NICE(
        prior=args.prior,
        coupling=args.coupling,
        coupling_type=args.coupling_type,
        in_out_dim=args.full_dim,
        mid_dim=args.mid_dim,
        hidden=args.hidden,
        device=device).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(), lr=args.lr)

    train_log_likelihood, test_log_likelihood = [], []
    # TODO fill in
    for e in tqdm.tqdm(range(args.epochs)):
        train_result = train(flow, trainloader, optimizer, e)
        test_result = test(flow, testloader, model_save_filename, e, sample_shape)
        train_log_likelihood.append(train_result)
        test_log_likelihood.append(test_result)

    plot_draws(train_log_likelihood, test_log_likelihood)

    print('Finished Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--dataset',
                        help='dataset to be modeled.',
                        type=str,
                        default='mnist')
    parser.add_argument('--prior',
                        help='latent distribution.',
                        type=str,
                        default='logistic')
    parser.add_argument('--full_dim',
                        help='full dim.',
                        type=int,
                        default=28 * 28)
    parser.add_argument('--batch_size',
                        help='number of images in a mini-batch.',
                        type=int,
                        default=128)
    parser.add_argument('--epochs',
                        help='maximum number of iterations.',
                        type=int,
                        default=50)
    parser.add_argument('--sample_size',
                        help='number of images to generate.',
                        type=int,
                        default=64)
    parser.add_argument('--coupling-type',
                        help='.',
                        type=str,
                        default='affine')
    parser.add_argument('--coupling',
                        help='.',
                        # type=int,
                        default=4)
    parser.add_argument('--mid-dim',
                        help='.',
                        type=int,
                        default=1000)
    parser.add_argument('--hidden',
                        help='.',
                        type=int,
                        default=5)
    parser.add_argument('--lr',
                        help='initial learning rate.',
                        type=float,
                        default=2e-4)

    args = parser.parse_args()
    main(args)
