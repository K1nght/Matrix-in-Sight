import time
import logging
from utils.log import Logger
from det_numbers.calc_acc import *
from det_numbers.datasets_new import *
from det_numbers.num_model import *
from det_numbers.datasets_mnistv2 import *
from tqdm import tqdm


def train(model, optimizer, epoch, dataloader):
    logger.info("        =======  start  training   ======     ")
    model.train()
    loss_mean = 0
    acc_mean = 0
    with tqdm(dataloader) as pbar:
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()

            optimizer.zero_grad()
            output = model(data)

            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss.backward()
            optimizer.step()

            loss = loss.item()
            acc = calc_acc(target, output)

            if batch_index == 0:
                loss_mean = loss
                acc_mean = acc

            loss_mean = 0.1 * loss + 0.9 * loss_mean
            acc_mean = 0.1 * acc + 0.9 * acc_mean

            pbar.set_description(f'Train:batch_id: {batch_index} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
    logger.info('Train: Epoch: %d Loss: %.4f Acc: %.4f'%(epoch, loss_mean, acc_mean))
    logger.info("        =======  finish  training   ======     ")


def valid(model, optimizer, epoch, dataloader):
    logger.info("        =======  start  validing   ======     ")
    model.eval()
    with tqdm(dataloader) as pbar, torch.no_grad():
        loss_sum = 0
        acc_sum = 0
        for batch_index, (data, target, input_lengths, target_lengths) in enumerate(pbar):
            data, target = data.cuda(), target.cuda()

            output = model(data)
            output_log_softmax = F.log_softmax(output, dim=-1)
            loss = F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)

            loss = loss.item()
            acc = calc_acc(target, output)

            loss_sum += loss
            acc_sum += acc

            loss_mean = loss_sum / (batch_index + 1)
            acc_mean = acc_sum / (batch_index + 1)

            pbar.set_description(f'Test:batch_id: {batch_index} Loss: {loss_mean:.4f} Acc: {acc_mean:.4f} ')
    logger.info('Test: Epoch: %d Loss: %.4f Acc: %.4f'%(epoch, loss_mean, acc_mean))
    logger.info("        =======  finish  validing   ======     ")


if __name__ == '__main__':
    save_dir = "checkpoint"
    save_path = os.path.join(save_dir, time.strftime("%b-%d[%H.%M.%S]-", time.localtime()) + "NUM")
    os.makedirs(save_path)
    logger = Logger(
        log_file_name=save_path + "/log.txt",
        log_level=logging.DEBUG,
        logger_name="NUM",
    ).get_log()
    logger.info("        =======  Training Setting   ======     ")
    logger.info("characters[%s]" % characters)
    logger.info("width[%d]|height[%d]|n_len[%d]|n_classes[%d]" % (width, height, n_len, n_classes))
    logger.info("n_len[%d]|n_classes[%d]" % (n_len, n_classes))
    logger.info("n_input_length[%d]" % n_input_length)
    logger.info("        =======  Training Setting   ======     ")

    # 初始化数据集生成器
    batch_size = 1
    train_set = CaptchaDataset(characters, n_input_length, n_len)
    valid_set = CaptchaDataset(characters, n_input_length, n_len, 1000)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=2)
    # 初始化模型
    width, height = 192, 64,
    model = Model(n_classes, input_shape=(3, height, width))
    model = model.cuda()
    # 开始训练
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=True)
    epochs = 4
    for epoch in range(1, epochs + 1):
        train(model, optimizer, epoch, train_loader)
        valid(model, optimizer, epoch, valid_loader)
        save_model_path = os.path.join(save_path, 'num-%d.pth' % epoch)
        torch.save(model, save_model_path)
    save_model_path = os.path.join(save_path, 'num-last.pth')
    torch.save(model, save_model_path)
