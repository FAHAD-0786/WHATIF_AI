"""Encryption vault.

Your passphrase is turned into a key with scrypt (slow on purpose, so guessing is hard).
Every piece of diary text is sealed with AES-256-GCM. The key lives only in RAM and is
never written to disk. Forget the passphrase = the data can never be recovered.
"""
import hashlib
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

_CHECK = b"whatif-vault-ok"


class WrongPassphrase(Exception):
    pass


class Vault:
    def __init__(self, folder):
        self.salt_file = folder / "vault.salt"
        self.check_file = folder / "vault.check"
        self._aes = None

    @property
    def initialised(self):
        return self.salt_file.exists() and self.check_file.exists()

    @property
    def unlocked(self):
        return self._aes is not None

    def unlock(self, passphrase: str):
        if len(passphrase) < 8:
            raise ValueError("Passphrase must be at least 8 characters.")
        first_time = not self.initialised
        salt = os.urandom(16) if first_time else self.salt_file.read_bytes()
        key = hashlib.scrypt(passphrase.encode(), salt=salt, n=2**15, r=8, p=1,
                             dklen=32, maxmem=128 * 1024 * 1024)
        aes = AESGCM(key)
        if first_time:
            self.salt_file.write_bytes(salt)
            self.check_file.write_bytes(self._seal(aes, _CHECK))
        else:
            try:
                if self._open(aes, self.check_file.read_bytes()) != _CHECK:
                    raise WrongPassphrase()
            except InvalidTag:
                raise WrongPassphrase()
        self._aes = aes

    def lock(self):
        self._aes = None

    @staticmethod
    def _seal(aes, data):
        nonce = os.urandom(12)
        return nonce + aes.encrypt(nonce, data, None)

    @staticmethod
    def _open(aes, blob):
        return aes.decrypt(blob[:12], blob[12:], None)

    def encrypt(self, data: bytes) -> bytes:
        return self._seal(self._aes, data)

    def decrypt(self, blob: bytes) -> bytes:
        return self._open(self._aes, blob)
