# ============================================================
# 【模块说明】命令行微调训练入口（适合 IP102 等大规模害虫数据集）
#
#   小程序的「训练」页是通过 base64 一张张上传图片的，只适合每类几十张的小样本；
#   IP102 这类几万张的数据集请把孩子目录直接放进数据集根目录，再用本脚本训练。
#
# 用法：
#   cd backend/app
#   uv run python train_cli.py --dry-run               # 只统计数据集，不训练
#   uv run python train_cli.py --epochs 10             # 用默认数据集目录训练
#   uv run python train_cli.py --data D:\ip102_subset --epochs 20
#
# 数据集目录结构（子目录名 = 类别名，会直接成为识别结果里的类别）：
#   data/upload/dataset/unzipped/
#       ├── 二化螟/  *.jpg
#       ├── 稻纵卷叶螟/ *.jpg
#       └── 粘虫/    *.jpg
# ============================================================
import argparse
import sys
from pathlib import Path

# 与 main.py 相同的路径处理：把 backend/ 和 backend/app/ 加入模块搜索路径
_APP_DIR = Path(__file__).resolve().parent
_BACKEND_DIR = _APP_DIR.parent
sys.path.insert(0, str(_BACKEND_DIR))
sys.path.insert(0, str(_APP_DIR))

from config.config import settings
from utils.common_utils import default_logger
from services.classify_service import ClassifyService

# 图片扩展名（与 torchvision ImageFolder 支持范围保持一致）
IMAGE_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}


def scan_dataset(data_dir: Path) -> dict:
    """统计数据集：类别数与每个类别的图片数（用于训练前自检数据是否就位）"""
    stats = {}
    if not data_dir.exists():
        return stats
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        n = sum(1 for f in sub.rglob('*') if f.suffix.lower() in IMAGE_EXT)
        stats[sub.name] = n
    return stats


def main() -> int:
    """解析参数 → 打印数据集概况 → 调用识别服务的微调流程 → 返回退出码"""
    parser = argparse.ArgumentParser(description='农田害虫识别模型微调（命令行）')
    parser.add_argument('--data', default=settings.UPLOAD_DATASET_UNZIPED_DIR,
                        help='数据集根目录（每个子目录 = 一个害虫类别）')
    parser.add_argument('--epochs', type=int, default=settings.FULL_EPOCHS,
                        help='训练轮数（默认取 config.FULL_EPOCHS）')
    parser.add_argument('--dry-run', action='store_true',
                        help='只统计并打印数据集概况，不真正训练')
    parser.add_argument('--unfreeze', action='store_true',
                        help='同时解冻 layer4 微调（准确率更高，耗时更长）')
    args = parser.parse_args()

    data_dir = Path(args.data)
    stats = scan_dataset(data_dir)

    print('=' * 60)
    print('数据集目录：%s' % data_dir)
    print('类别数    ：%d' % len(stats))
    total = sum(stats.values())
    print('图片总数  ：%d' % total)
    for name, n in stats.items():
        print('    %-30s %5d 张' % (name, n))
    print('设备      ：%s（GPU 层数 %s）' % (settings.DEVICE, settings.GPU_MEMORY_THRESHOLD_GB))
    print('=' * 60)

    # 数据自检：ImageFolder 要求至少 2 个类别才有意义（单类别无法区分）
    if len(stats) < 2:
        default_logger.error('数据集不足：至少需要 2 个类别目录，且每个目录内放图片')
        return 1
    empty = [k for k, v in stats.items() if v == 0]
    if empty:
        default_logger.warning('以下类别没有图片，将被忽略：%s' % ', '.join(empty))

    if args.dry_run:
        print('dry-run 模式：仅统计，不训练')
        return 0

    print('开始微调，共 %d 个 epoch …' % args.epochs)
    svc = ClassifyService()                    # 单例：内部会先加载模型
    ok = svc.fintune(date_dir=data_dir, epoch=args.epochs, unfreeze=args.unfreeze)
    print('训练结果：%s' % ('成功（模型已保存）' if ok else '失败（详见日志）'))
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
