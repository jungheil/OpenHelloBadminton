# -*- coding: utf-8 -*-

# File: passwd_manager.py
# License: MIT License
# Copyright: (c) 2023 Jungheil <jungheilai@gmail.com>
# Created: 2023-11-03
# Brief:
# --------------------------------------------------

import base64
import hashlib
import pickle
from pathlib import Path
from typing import Optional, Tuple

from cryptography.fernet import Fernet, InvalidToken

from hello_badminton.utils.utils import AccountType


class PasswdManager:
    def __init__(
        self,
        passwd_file: str,
    ) -> None:
        self._passwd_file = Path(passwd_file)
        self._cipher_suite = None
        self._data = {}

    @property
    def data(self) -> dict:
        return self._data

    def _init_passwd_file(self):
        if not self._passwd_file.exists():
            self._passwd_file.parent.mkdir(parents=True, exist_ok=True)
            self._save_data()

    def login(self, key) -> bool:
        key = base64.b64encode(hashlib.sha256(key.encode()).digest())
        self._cipher_suite = Fernet(key)
        self._init_passwd_file()
        try:
            self._load_passwd()
            return True
        except InvalidToken:
            self._cipher_suite = None
            return False

    def _save_data(self) -> None:
        if self._cipher_suite is None:
            raise ValueError("Please login first!")
        pickled_data = pickle.dumps(self._data)
        encrypted_data = self._cipher_suite.encrypt(pickled_data)
        with open(self._passwd_file.absolute(), "wb") as f:
            f.write(encrypted_data)

    def _load_passwd(self) -> None:
        if self._cipher_suite is None:
            raise ValueError("Please login first!")
        with open(self._passwd_file.absolute(), "rb") as f:
            encrypted_data = f.read()
        decrypted_data = self._cipher_suite.decrypt(encrypted_data)
        self._data = pickle.loads(decrypted_data)

    def get_account(self, username: str) -> Tuple[Optional[str], Optional[AccountType]]:
        data = self._data.get(username)
        if data is not None:
            return data[0], AccountType(data[1])
        return None, None

    def set_account(self, username: str, passwd: str, acc_type: AccountType) -> None:
        self._data[username] = (passwd, acc_type.value)
        self._save_data()

    def del_account(self, username: str) -> bool:
        try:
            self._data.pop(username)
        except KeyError:
            return False
        self._save_data()
        return True
