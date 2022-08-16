// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Diagnostics;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;

namespace CorrectionWebApp.Helper
{
    public static class ShellHelper
    {
        public static Task<int> Bash(this string cmd, ILogger logger)
        {
            var source = new TaskCompletionSource<int>();
            var escapedArgs = cmd.Replace("\"", "\\\"");
            var process = new Process
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "bash",
                    Arguments = $"-c \"{escapedArgs}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true
                },
                EnableRaisingEvents = true
            };
            process.Exited += (sender, args) =>
            {
                logger.LogWarning(process.StandardError.ReadToEnd());
                logger.LogInformation(process.StandardOutput.ReadToEnd());
                if (process.ExitCode == 0)
                {
                    source.SetResult(0);
                }
                else
                {
                    source.SetException(new Exception($"Command `{cmd}` failed with exit code `{process.ExitCode}`"));
                }
                process.Dispose();
            };
            try
            {
                process.Start();
            }
            catch (Exception e)
            {
                logger.LogError(e, "Command {} failed", cmd);
                source.SetException(e);
            }
            return source.Task;
        }


        public static async Task<string> GetBash(this string cmd, ILogger logger)
        {
            var result = "";
            var source = new TaskCompletionSource<int>();
            var escapedArgs = cmd.Replace("\"", "\\\"");
            var process = new Process()
            {
                StartInfo = new ProcessStartInfo
                {
                    FileName = "bash",
                    Arguments = $"-c \"{escapedArgs}\"",
                    RedirectStandardOutput = true,
                    RedirectStandardError = true,
                    UseShellExecute = false,
                    CreateNoWindow = true,
                },
                EnableRaisingEvents = true
            };
            process.Exited += (sender, args) =>
            {
                // logger.LogWarning(process.StandardError.ReadToEnd());
                result = process.StandardOutput.ReadToEnd();
                logger.LogWarning(result);
                logger.LogInformation(process.StandardOutput.ReadToEnd());
                if (process.ExitCode == 0)
                {
                    source.SetResult(0);
                }
                else
                {
                    source.SetException(new Exception($"Command `{cmd}` failed with exit code `{process.ExitCode}`"));
                }
                process.Dispose();
            };
            try
            {
                process.Start();
            }
            catch (Exception e)
            {
                logger.LogError(e, "Command {} failed", cmd);
                source.SetException(e);
            }

            await process.WaitForExitAsync();
            return result;
        }
    }
}

