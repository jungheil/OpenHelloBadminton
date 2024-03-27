# -*- coding: utf-8 -*-

# File: main.py
# License: MIT License
# Copyright: (c) 2023 Jungheil <jungheilai@gmail.com>
# Created: 2023-11-03
# Brief:
# --------------------------------------------------

import getpass
import json

import click

from hello_badminton import HelloBadminton, PasswdManager
from hello_badminton.utils.utils import AccountType


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--config", type=click.File("r"), default="./config.json", help="Config file path."
)
def run(config):
    param = json.load(config)
    hello_badminton = HelloBadminton(param)
    hello_badminton()


@cli.command()
def pm():
    pm = PasswdManager("data/passwd")
    while True:
        key = getpass.getpass("Please input your key: ")
        if pm.login(key):
            break
        print("Wrong key!")
    while True:
        cmd = input(
            "Please input your command. 1 for get, 2 for set, 3 for delete, 0 for exit:\n> "
        )
        if cmd == "1":
            data = pm.data.items()
            if len(data):
                for k, v in data:
                    print(
                        f"username: {k};\tpassword: {v[0]};\ttype: {AccountType(v[1]).name}"
                    )
            else:
                print("No data!")
        elif cmd == "2":
            username = input("Please input your username:\n> ")
            passwd = getpass.getpass("Please input your password:\n> ")
            acc_type = input(
                f"Please input your account type. {', '.join([f'{i.value} for {i.name}' for i in AccountType])}:\n> "
            )
            try:
                acc_type = AccountType(int(acc_type))
            except ValueError:
                print("Invalid account type!")
                continue
            pm.set_account(username, passwd, acc_type)
            print("Set successfully!")
        elif cmd == "3":
            username = input("Please input your username:\n> ")
            if pm.del_account(username):
                print("Delete successfully!")
            else:
                print("Delete failed!")
        elif cmd == "0":
            print("Bye!")
            break
        else:
            print("Invalid command!")


if __name__ == "__main__":
    cli()
