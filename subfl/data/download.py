"""Open a torchvision dataset, downloading it only when the config allows."""

from __future__ import annotations


def open_dataset(cls, config, **kwargs):
    """Construct the torchvision dataset ``cls`` under ``config.root``, downloading only if ``config.download``."""

    root = str(config.root)
    download = bool(config.download)
    try:
        return cls(root=root, download=download, **kwargs)
    except RuntimeError as exc:
        if download:
            raise
        raise RuntimeError(f"{cls.__name__} not found or corrupt under {root} and downloading is off: fix "
                           f"--data_root, or pass --download_data to let torchvision download it ({exc})") from exc
