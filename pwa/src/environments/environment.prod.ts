// Copyright (C) 2020 - 2022 APC Inc.

export const environment = {
  production: true,
  apiBaseURL: 'http://localhost:5012',
  mqtt: {server: 'localhost', protocol: 'wss', port: 9001, prefixTopic: "prefix-topic"},
  expires: 8
};
