## Hello Badminton

```
  _    _      _ _       ____            _           _       _
 | |  | |    | | |     |  _ \          | |         (_)     | |
 | |__| | ___| | | ___ | |_) | __ _  __| |_ __ ___  _ _ __ | |_ ___  _ __
 |  __  |/ _ \ | |/ _ \|  _ < / _` |/ _` | '_ ` _ \| | '_ \| __/ _ \| '_ \
 | |  | |  __/ | | (_) | |_) | (_| | (_| | | | | | | | | | | || (_) | | | |
 |_|  |_|\___|_|_|\___/|____/ \__,_|\__,_|_| |_| |_|_|_| |_|\__\___/|_| |_|

```

这是一个为了解决因爬虫泛滥导致无法正常在**XX大学场馆管理系统**中预约体育场地的问题而生的项目。**（为防止滥用已删除关键代码文件后开源）**

### Environment and Dependence

- python 3.11
- node 12.22

```bash
pip install -r requirements.txt
npm install crypto-js
```

### Run

1. 更改 `config.json`中的设置

2. 设置主、同伴帐号与密码

   ```bash
   python main.py pm
   ```

3. 执行程序

   ```bash
   python main.py run [--config <config_file_path>.json]
   ```

### Dist

```bash
python setup.py
```

### Config

其中标注\*为可空

| name                 | remark                                              |
| -------------------- | --------------------------------------------------- |
| `venue_site_id`      | 场馆编号                                            |
| `reservation_date`\* | 预约时间，格式为 `%Y-%m-%d`，为空则为执行时间两天后 |
| `main_user`\*        | 主帐号，需要在`passwd_manager.py`中添加密码         |
| `buddy_user`\*       | 同伴帐号，需要在`passwd_manager.py`中添加密码       |
| `phone`              | 电话号码                                            |
| `exec_time`\*        | 执行时间，格式为 `%Y-%m-%d %H:%M:%S`，空则立刻执行  |
| `exec_delay`         | 执行延迟时间，单位 s                                |
| `max_try`            | 最大尝试次数                                        |
| `max_try_captcha`    | 最大验证码尝试次数                                  |
| `cdd_time`\*         | 备选时间，格式为 `beginTime-endTime`                |
| `cdd_venue`\*        | 备选场地                                            |

### Milestone

| feature             | status                   |
| ------------------- | ------------------------ |
| **系统 api 接口**   | 🎉️                        |
| 执行流程与人机交互  | 👌🏻️                        |
| 手点验证码          | 👌🏻️                        |
| **自动验证码**      | 🎉️ 基于孪生网络           |
| 验证码提前          | 👌🏻️                        |
| `Pyppeteer`自动登录 | 🎉️                        |
| 密码密文储存        | 👌🏻️                        |
| api 登录            | 😭️ 估计不会有，等一个大佬 |
| **第一次成功预约**  | 🎉️ 2023.11.15             |
| **并发请求**        | 🎉️                        |
| 程序编译            | 👌🏻️                        |
| 发现与修复 BUG      | 🧑🏻‍🦲 持续进行中...         |

### Q&A

- 忘记本地密码存储的`key`?

  删除`data/passwd`并重新存储帐号与密码。

### Acknowledgment

- <https://github.com/pnpnpn/timeout-decorator>
- <https://github.com/bubbliiiing/Siamese-pytorch>
- <https://github.com/sml2h3/ddddocr>
