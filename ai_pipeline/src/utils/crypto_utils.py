from cryptography.fernet import Fernet


def encrypt_bytes(key: bytes, data: bytes) -> bytes:
    return Fernet(key).encrypt(data)


def decrypt_bytes(key: bytes, token: bytes) -> bytes:
    return Fernet(key).decrypt(token)
