# Copyright (c) 2020, G.A. vd. Hoorn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# author: G.A. vd. Hoorn


import ftplib

from io import BytesIO


class FtpClient(object):
    def __init__(self, host: str, timeout: float = 5) -> None:
        self.host = host
        self.ftpc = ftplib.FTP(host, timeout=timeout)
        self.__welcome_msg = self.ftpc.getwelcome()

    def get_welcome_msg(self) -> str:
        return self.__welcome_msg

    def connect(self, user: str = 'anonymous', pw: str = 'anonymous') -> None:
        self.ftpc.login(user, pw)

    def get_file_as_str(self, remote_name: str) -> bytes:
        buf = BytesIO()
        self.ftpc.retrbinary(f'RETR {remote_name}', buf.write)
        return buf.getvalue()

    def upload_as_file(self, remote_name: str, contents: bytes) -> None:
        buf = BytesIO(contents)
        buf.seek(0)
        self.ftpc.storbinary(f'STOR {remote_name}', buf)

    def remove_file(self, remote_name: str) -> None:
        self.ftpc.delete(remote_name)
