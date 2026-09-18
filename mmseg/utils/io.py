import gzip
import io
import pickle

import cv2
import numpy as np


def datafrombytes(content: bytes, backend: str = 'numpy') -> np.ndarray:
    """Decode serialized array data used by MMSegmentation transforms."""
    if backend == 'pickle':
        return pickle.loads(content)

    with io.BytesIO(content) as file_obj:
        if backend == 'numpy':
            return np.load(file_obj)
        if backend == 'cv2':
            encoded = np.frombuffer(file_obj.read(), dtype=np.uint8)
            return cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if backend == 'nifti':
            file_obj = gzip.open(file_obj)
            from nibabel import FileHolder, Nifti1Image
            holder = FileHolder(fileobj=file_obj)
            image = Nifti1Image.from_file_map({
                'header': holder,
                'image': holder
            })
            return Nifti1Image.from_bytes(image.to_bytes()).get_fdata()
    raise ValueError(f'Unsupported decoding backend: {backend}')
