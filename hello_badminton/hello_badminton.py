# -*- coding: utf-8 -*-

# File: hello_badminton.py
# License: MIT License
# Copyright: (c) 2023 Jungheil <jungheilai@gmail.com>
# Created: 2023-11-03
# Brief:
# --------------------------------------------------

import asyncio
import datetime
import functools
import getpass
import logging
import random
import sched
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import execjs
import tqdm
from curl_cffi import CurlError, requests

from hello_badminton.captcha import captcha_registry
from hello_badminton.hb_config import HBConfig
from hello_badminton.login_zju import login_auto_local, login_auto_sso, login_manual
from hello_badminton.passwd_manager import PasswdManager
from hello_badminton.tyys_api import (
    check_captcha,
    check_reservation,
    get_buddy_no,
    get_captcha,
    get_reservation,
    pay,
    solve_captcha,
    submit,
)
from hello_badminton.utils.utils import AccountType


def login(
    username: Optional[str] = None,
    passwd: Optional[str] = None,
    acc_type: Optional[AccountType] = None,
) -> dict:
    if username is None or passwd is None or acc_type is None:
        return asyncio.run(login_manual())
    else:
        while True:
            try:
                if acc_type == AccountType.SSO:
                    return asyncio.run(login_auto_sso(username, passwd))
                elif acc_type == AccountType.LOCAL:
                    return asyncio.run(login_auto_local(username, passwd))
                else:
                    raise ValueError("Invalid account type.")
            except TimeoutError:
                print("Login timeout! Retrying...")


def get_buddy_by_login(
    username: Optional[str] = None,
    passwd: Optional[str] = None,
    acc_type: Optional[AccountType] = None,
) -> tuple:
    login_data = login(username, passwd, acc_type)
    buddy_no = asyncio.run(get_buddy_no(login_data=login_data))
    return (str(login_data["userid"]), buddy_no)


def get_server_delay() -> float:
    def _get_server_delay():
        local_time = time.time()
        response = requests.head(
            "http://tyys.zju.edu.cn/venue-server/api/reservation/order/submit",
            verify=False,
            impersonate="chrome110",
        )
        server_time = execjs.eval(
            f"new Date('{response.headers['Date']}').getTime() / 1000"
        )
        return server_time - local_time

    try:
        delay_list = [_get_server_delay() for _ in range(10)]
        return sum(delay_list) / len(delay_list)
    except TimeoutError:
        print("Failed to get server delay. Use default delay 0s.")
        return 0


def _check_dependence():
    execjs.eval(r"CryptoJS = require('crypto-js'),CryptoJS.enc.Utf8.parse('t')")


def _check_config(conf: HBConfig):
    assert (
        conf.solve_captcha in captcha_registry.captcha.keys()
    ), f"solve_captcha should be in {captcha_registry.captcha.keys()}"


class HelloBadminton:
    def __init__(self, param: dict) -> None:
        self._conf = HBConfig(**param)
        _check_dependence()
        _check_config(self._conf)

        Path("logs").mkdir(exist_ok=True)
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        )
        self._logger.addHandler(stream_handler)
        file_handler = logging.FileHandler(
            f"logs/{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}.log"
        )
        file_handler.setFormatter(
            logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        )
        self._logger.addHandler(file_handler)

        self._pm = PasswdManager("data/passwd")

        self._login_data = None
        self._captcha_verification = []
        self._buddy = []
        self._status = False

    def __call__(self):
        self._logger.info("Hello, dear DDL!")
        self._logger.info("Config: \n%s", self._conf)
        self._logger.info(
            "CAUTION: You need to ensure that you have MANUALLY added the buddy account to the main account."
        )
        while True:
            confirm = input("Are you sure? (Y/n)\n> ")
            if confirm.lower() == "n":
                self._logger.info("Bye!")
                exit(0)
            elif confirm.lower() == "y" or len(confirm) == 0:
                break
        if self._conf.main_user is not None or self._conf.buddy_user is not None:
            while True:
                key = getpass.getpass("Please input your key: ")
                if self._pm.login(key):
                    break
                self._logger.warning("Wrong key!")

        if self._conf.exec_time is not None and self._conf.login_ahead_time is not None:
            login_time = self._conf.exec_time - datetime.timedelta(
                seconds=self._conf.login_ahead_time
            )
            delay = login_time.timestamp() - time.time()
            if delay > 0:
                self._logger.info("Scheduled to run at %s", login_time)
                time.sleep(delay)

        self._logger.info("Logging in...")
        try:
            passwd, acc_type = self._pm.get_account(self._conf.main_user)
            self._login_data = login(self._conf.main_user, passwd, acc_type)
        except Exception as e:
            self._logger.error("Login failed.")
            tb_info = traceback.extract_tb(e.__traceback__)
            for tb in tb_info:
                filename, line, func, text = tb
                self._logger.error(
                    "File: %s, Line: %s, Function: %s, Text: %s",
                    filename,
                    line,
                    func,
                    text,
                )
            raise RuntimeError("Login failed.") from e
        self._logger.info("Login success.")

        self.get_buddy(self._conf.buddy_user)
        self.select_cdd()

        if self._conf.exec_time is None:
            self.reserve()
        else:
            self._reserve_scheduler()

        for _ in range(self._conf.max_try - 1):
            if self._status:
                break
            self.reserve()

        self._logger.info("Bye!")

    def _reserve_scheduler(self):
        now_time = time.time()
        pbar = tqdm.tqdm(
            total=round(self._get_exec_time() - now_time, 2),
            desc="Waiting",
            bar_format=" {desc}: {percentage:3.0f}%|{bar}| {n:.2f}s/{total:.2f}s",
        )

        def _countdown(
            scheduler: sched.scheduler, last_time: float, freq: float = 1.0
        ) -> None:
            now_time = time.time()
            if self._get_exec_time() - now_time > 1:
                pbar.update(now_time - last_time)
                scheduler.enter(freq, 100, _countdown, (scheduler, now_time))
            else:
                pbar.close()

        scheduler = sched.scheduler(time.time, time.sleep)
        if self._conf.captcha_ahead_time is not None:
            scheduler.enterabs(
                self._get_exec_time() - self._conf.captcha_ahead_time,
                5,
                self.pre_captcha,
                (self._conf.max_run, self._conf.captcha_ahead_time),
            )
        scheduler.enterabs(self._get_exec_time(), 0, self.reserve)
        scheduler.enter(0, 100, _countdown, (scheduler, now_time))

        self._logger.info("ready to reserve at %s", self._conf.exec_time)
        scheduler.run()

    def pre_captcha(self, count=1, timeout=None):
        async def _pre_captcha():
            tasks = [self.get_captcha_verification(timeout) for _ in range(count)]
            fut = asyncio.gather(*tasks)
            self._captcha_verification.extend([i for i in await fut if i is not None])

        asyncio.run(_pre_captcha())

    @functools.lru_cache(maxsize=1)
    def _get_exec_time(self):
        exec_timestamp = self._conf.exec_time.timestamp()
        delay = get_server_delay() + self._conf.exec_delay
        exec_timestamp += delay
        return exec_timestamp

    def select_cdd(self):
        reservation_date = datetime.datetime.now().strftime("%Y-%m-%d")
        argument = {
            "venueSiteId": self._conf.venue_site_id,
            "reservationDate": reservation_date,
        }
        reservation = asyncio.run(get_reservation(self._login_data, **argument))
        if reservation["code"] != 200:
            self._logger.error("Failed to get reservation info. Got %s", reservation)
            raise RuntimeError("Failed to get reservation info.")

        all_time = dict(
            [
                (i, f'{info["beginTime"]}-{info["endTime"]}')
                for i, info in enumerate(reservation["data"]["spaceTimeInfo"])
            ]
        )
        if self._conf.cdd_time is not None and (
            all(i in all_time.values() for i in self._conf.cdd_time)
            or self._logger.warning("Invalid time. Please input again.")
            or False
        ):
            pass
        else:
            self._logger.info(all_time)
            while True:
                input_time = input(
                    "Please input the time id you want to reserve, e.g. 0,1,6\n> "
                )
                cdd_time_id = input_time.split(",")
                try:
                    self._conf.cdd_time = [all_time[int(i)] for i in cdd_time_id]
                    break
                except KeyError as e:
                    print(e)
                    self._logger.error("Invalid time id. Please input again.")
            self._logger.info("Time selected.")
        self._logger.info("cdd_time: %s", self._conf.cdd_time)

        all_venue = dict(
            [
                (i, info["spaceName"])
                for i, info in enumerate(
                    reservation["data"]["reservationDateSpaceInfo"][reservation_date]
                )
            ]
        )
        if len(self._conf.cdd_venue) != 0 and (
            all(i in all_venue.values() for i in self._conf.cdd_venue)
            or self._logger.warning("Invalid venue. Please input again.")
            or False
        ):
            pass
        else:
            self._logger.info(all_venue)

            while True:
                input_venue = input(
                    "Please input the venue id you want to reserve, e.g. 0,1,6\n> "
                )
                cdd_venue = input_venue.split(",")
                try:
                    self._conf.cdd_venue = [all_venue[int(i)] for i in cdd_venue]
                    break
                except KeyError:
                    self._logger.error("Invalid venue id. Please input again.")
            self._logger.info("Venue selected.")
        self._logger.info("cdd_venue: %s", self._conf.cdd_venue)

    def get_buddy(self, buddy_user: Optional[List]):
        self._logger.info("Adding buddy...")
        if buddy_user is not None:
            for user in buddy_user:
                try:
                    passwd, acc_type = self._pm.get_account(user)
                    ret = get_buddy_by_login(user, passwd, acc_type)
                    self._buddy.append(ret)
                except Exception as e:
                    self._logger.warning("Failed to get buddy. Got %s", e)
                    continue

        if len(self._buddy) == 0:
            self._logger.info("Please login to get buddy id.")
            while True:
                try:
                    ret = get_buddy_by_login()
                    self._buddy.append(ret)
                    self._logger.info("Added buddy %s, buddy id %s.", ret[0], ret[1])
                    assert len(ret[1]) > 0, "Get buddy id failed."
                    break
                except AssertionError as e:
                    self._logger.warning("Failed to get buddy. Got %s", e)
                    tb_info = traceback.extract_tb(e.__traceback__)
                    for tb in tb_info:
                        filename, line, func, text = tb
                        self._logger.warning(
                            "File: %s, Line: %s, Function: %s, Text: %s",
                            filename,
                            line,
                            func,
                            text,
                        )
            while True:
                input_buddy = input("Do you want to add another buddy? (y/N)\n> ")
                if input_buddy.lower() == "n" or len(input_buddy) == 0:
                    break
                elif input_buddy.lower() == "y":
                    try:
                        ret = get_buddy_by_login()
                        self._buddy.append(ret)
                        self._logger.info(
                            "Added buddy %s, buddy id %s.", ret[0], ret[1]
                        )
                        assert len(ret[1]) > 0, "Get buddy id failed."
                    except AssertionError as e:
                        self._logger.warning("Failed to get buddy. Got %s", e)
                        tb_info = traceback.extract_tb(e.__traceback__)
                        for tb in tb_info:
                            filename, line, func, text = tb
                            self._logger.warning(
                                "File: %s, Line: %s, Function: %s, Text: %s",
                                filename,
                                line,
                                func,
                                text,
                            )
                else:
                    continue

        self._logger.info("Add Buddy finished.")
        self._logger.info("Buddy info: %s", self._buddy)

    def get_available_venue(self, data):
        time_id_map = dict(
            (
                (f'{i["beginTime"]}-{i["endTime"]}', str(i["id"]))
                for i in data["data"]["spaceTimeInfo"]
            )
        )

        venue_id_map = dict(
            (
                (i["spaceName"], str(i["id"]))
                for i in data["data"]["reservationDateSpaceInfo"][
                    self._conf.reservation_date.strftime("%Y-%m-%d")
                ]
            )
        )

        cdd_time_id = [time_id_map[t] for t in self._conf.cdd_time if t in time_id_map]
        cdd_venue_id = [
            venue_id_map[v] for v in self._conf.cdd_venue if v in venue_id_map
        ]

        cdd_available_name = [
            (str(i["spaceName"]), tn)
            for tn, td in time_id_map.items()
            if td in cdd_time_id
            for i in data["data"]["reservationDateSpaceInfo"][
                self._conf.reservation_date.strftime("%Y-%m-%d")
            ]
            if str(i["id"]) in cdd_venue_id and i[td]["reservationStatus"] == 1
        ]

        time_idx = dict([(v, i) for i, v in enumerate(self._conf.cdd_time)])
        venue_idx = dict([(v, i) for i, v in enumerate(self._conf.cdd_venue)])
        cdd_available_name = sorted(
            cdd_available_name, key=lambda x: (time_idx[x[1]], venue_idx[x[0]])
        )

        self._logger.info("Available venue: %s", cdd_available_name)

        cdd_available = [
            (venue_id_map[i[0]], time_id_map[i[1]]) for i in cdd_available_name
        ]

        return cdd_available, cdd_available_name

    async def get_captcha_verification(self, timeout: Optional[float] = None):
        begin_time = time.time()

        def _get_timeout():
            if timeout is None:
                return None
            else:
                return max(0, begin_time + timeout - time.time())

        self._logger.info("Getting captcha...")
        for _ in range(self._conf.max_try_captcha):
            self._logger.info("Trying to solve captcha.")
            try:
                captcha_data = await get_captcha(
                    self._login_data, timeout=_get_timeout()
                )
                captcha_result = await asyncio.to_thread(
                    solve_captcha,
                    captcha_data,
                    captcha_registry.get(self._conf.solve_captcha),
                    _get_timeout(),
                )
                if not captcha_result:
                    self._logger.warning("Failed to solve captcha. Try again.")
                    continue
                ret = await check_captcha(self._login_data, captcha_result)
                if (
                    ret.get("code") == 200
                    and ret.get("data") is not None
                    and ret.get("data").get("repCode") == "0000"
                ):
                    self._logger.info("Captcha solved.")
                    return captcha_result["captchaVerification"]
                else:
                    self._logger.warning("Failed to solve captcha.")
                    continue
            except TimeoutError as e:
                self._logger.warning("Get captcha timeout. Got %s", e)
                return None
            except CurlError as e:
                if e.code == 7:
                    self._logger.warning("Get captcha timeout. Got %s", e)
                    return None
                else:
                    self._logger.error("Failed to get captcha. Got %s", e)
                    tb_info = traceback.extract_tb(e.__traceback__)
                    for tb in tb_info:
                        filename, line, func, text = tb
                        self._logger.error(
                            "File: %s, Line: %s, Function: %s, Text: %s",
                            filename,
                            line,
                            func,
                            text,
                        )
            except Exception as e:
                self._logger.error("Failed to get captcha. Got %s", e)
                tb_info = traceback.extract_tb(e.__traceback__)
                for tb in tb_info:
                    filename, line, func, text = tb
                    self._logger.error(
                        "File: %s, Line: %s, Function: %s, Text: %s",
                        filename,
                        line,
                        func,
                        text,
                    )
        return None

    def process_buddy(self, data, argument):
        ids_dict = dict(
            [(str(i["userId"]), str(i["id"])) for i in data["data"]["buddyList"]]
        )
        try:
            buddy_ids = [ids_dict[i[0]] for i in self._buddy]
            buddy_no = [i[1] for i in self._buddy]
        except KeyError as e:
            self._logger.error("Please add buddy first.")
            raise RuntimeError("Failed to get buddy id.") from e
        argument["buddyIds"] = ",".join(buddy_ids)
        argument["buddyNo"] = ",".join(buddy_no)

    async def reserve_async(self) -> bool:
        self._logger.info("Reserve start.")
        argument_template = {
            "venueSiteId": self._conf.venue_site_id,
            "reservationDate": self._conf.reservation_date.strftime("%Y-%m-%d"),
            "phone": self._conf.phone,
        }
        available_id = []
        for _ in range(16):
            self._logger.info("Getting reservation info...")
            response = await get_reservation(self._login_data, **argument_template)
            if response.get("code") == 200:
                available_id, available_name = self.get_available_venue(response)
                break
            else:
                self._logger.warning("Failed to get reservation info. Got %s", response)
                continue
        begin_time = time.time()

        if len(available_id) == 0:
            self._logger.error("No available venue.")
            self._logger.info("Bye!")
            exit(0)
        available_id = available_id[: self._conf.max_run]
        available_name = available_name[: self._conf.max_run]

        tasks = []
        for idx, cdd in enumerate(zip(available_id, available_name)):
            id, name = cdd
            venue_id, time_id = id
            argument = argument_template.copy()
            argument["spaceId"] = venue_id
            argument["timeId"] = time_id
            try:
                argument["captchaVerification"] = self._captcha_verification.pop()
            except IndexError:
                pass
            self._logger.info("(task id: %s) Adding task %s.", idx, name)
            tasks.append(
                self.reserve_task(
                    idx,
                    argument,
                    response if idx == 0 else None,
                    begin_time if idx == 0 else None,
                )
            )
        ret = await asyncio.gather(*tasks)
        if any(ret):
            idx = next(i for i, v in enumerate(ret) if v)
            trade_data = ret[idx]
            self._logger.info("Paying...")
            pay_ret = await pay(self._login_data, **trade_data)
            if pay_ret["code"] == 200:
                self._logger.info("Pay success!")
                self._logger.info(
                    "Success! The trade No. is %s. The. The venue is %s.",
                    ret[idx]["tradeNo"],
                    available_name[idx],
                )

                return True
            else:
                self._logger.error("Pay failed. Got %s", pay_ret)
                self._logger.warning("Please pay MANUALLY in 5 minutes.")
                return False
        else:
            self._logger.info("Failed.")
            return False

    async def reserve_task(
        self, task_id, argument, rs_info=None, begin_time=None
    ) -> Dict[str, str]:
        self._logger.info("(task id: %s) Reserving...", task_id)
        try:
            self._logger.info("(task id: %s) Getting reservation info...", task_id)
            rs1 = (
                rs_info
                if rs_info is not None
                else await get_reservation(self._login_data, **argument)
            )
            begin_time = time.time() if begin_time is None else begin_time
            argument["token"] = rs1["data"]["token"]
            await asyncio.sleep(random.uniform(*self._conf.process_delay_range))
            self._logger.info("(task id: %s) Checking reservation info...", task_id)
            rs2 = await check_reservation(self._login_data, **argument)
            self.process_buddy(rs2, argument)
            await asyncio.sleep(random.uniform(*self._conf.process_delay_range))
            self._logger.info("(task id: %s) Captcha verification...", task_id)
            if argument.get("captchaVerification") is None:
                argument["captchaVerification"] = await self.get_captcha_verification()
            else:
                await asyncio.sleep(random.uniform(*self._conf.process_delay_range))
            delay_time = max(
                0.0, self._conf.min_process_time + begin_time - time.time()
            )
            self._logger.info(
                "(task id: %s) Waiting for %0.2fs...", task_id, delay_time
            )
            await asyncio.sleep(delay_time)
            self._logger.info("(task id: %s) Submitting...", task_id)
            ret = await submit(self._login_data, **argument)
            if ret["code"] == 200:
                self._logger.info("(task id: %s) reservation success!", task_id)
                self._logger.info(
                    "(task id: %s) Info: %s", task_id, ret["data"]["orderInfo"]
                )
                ret = {
                    "tradeId": ret["data"]["orderInfo"]["id"],
                    "tradeNo": ret["data"]["orderInfo"]["tradeNo"],
                }
                self._status = True
                return ret
            else:
                self._logger.error(
                    "(task id: %s) Something went wrong. Got %s", task_id, ret
                )
                return {}
        except Exception as e:
            self._logger.error("(task id: %s) Something went wrong. Got %s", task_id, e)
            tb_info = traceback.extract_tb(e.__traceback__)
            for tb in tb_info:
                filename, line, func, text = tb
                self._logger.error(
                    "File: %s, Line: %s, Function: %s, Text: %s",
                    filename,
                    line,
                    func,
                    text,
                )
            return {}

    def reserve(self) -> bool:
        return asyncio.run(self.reserve_async())
