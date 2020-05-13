# Copyright 2017 Rice University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

LABELS = ['swing', 'awt', 'security', 'sql', 'net', 'xml', 'crypto', 'math', 'lang', 'util']

def get_api(config, calls):

    apis = [config.vocab.chars_api[call] for call in calls]
    apis_ = []
    for api in apis:
        try:
            # print(api.split('.'))
            api_mid = api.split('.')[1]
        except:
            api_mid = []
        apis_.append(api_mid)

    guard = []
    for apis in apis_:
        if apis != []:
            if apis in LABELS:
                label = apis
                guard.append(label)

    if len(set(guard)) != 1:
        return 'N/A'
    else:
        return guard[0]