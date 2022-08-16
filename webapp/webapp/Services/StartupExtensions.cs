// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace CorrectionWebApp.Services
{
    public static class StartupExtensions
    {
        public static IServiceCollection AddAppServices(this IServiceCollection services, IConfiguration configuration, bool consoleApp = false)
        {
            #region add tlece db context
            var connectionString = configuration["ConnectionStrings:EntityFrameworkConnection"];
            var maxConnectionRetryCount = 0;
            var maxConnectionRetryDelaySeconds = 30;
            ICollection<int> transientSqlErrorNumbersToAdd = null;
            services.AddEntityFrameworkMySql()
                .AddDbContext<AppDbContext>(options =>
                    options.UseMySql(connectionString, new MySqlServerVersion(new Version(8, 0, 11)), //TODO confirm the correct version
                    mySqlOptionsAction: sqlOptions =>
                    {
                        if (maxConnectionRetryCount > 0)
                        {
                            //Configuring Connection Resiliency: https://docs.microsoft.com/en-us/ef/core/miscellaneous/connection-resiliency 
                            sqlOptions.EnableRetryOnFailure(
                                maxRetryCount: maxConnectionRetryCount,
                                maxRetryDelay: TimeSpan.FromSeconds(maxConnectionRetryDelaySeconds),
                                errorNumbersToAdd: transientSqlErrorNumbersToAdd);
                        }
                    }),
                    optionsLifetime: ServiceLifetime.Singleton
                    );
            #endregion

            services.TryAddScoped<AppService, AppService>();

            return services;
        }
    }
}
