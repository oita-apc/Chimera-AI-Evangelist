// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using CorrectionWebApp.Entities;
using Microsoft.EntityFrameworkCore;
using Microsoft.Extensions.Logging;

namespace CorrectionWebApp.Services
{
    public class AppDbContext : DbContext
    {
        protected AppDbContext(){}

        public AppDbContext(DbContextOptions<AppDbContext> options) : base(options) { }

        #region DbSet Definitions
        public virtual DbSet<AppImage> AppImages { get; set; }
        public virtual DbSet<AppImageAttribute> AppImageAttribute { get; set; }
        public virtual DbSet<AppAttribute> AppAttributes { get; set; }
        public virtual DbSet<AppAttributeCategory> AppAttributeCategory { get; set; }
        public virtual DbSet<AppUser> AppUsers { get; set; }
        public virtual DbSet<AppTrainingHistory> AppTrainingHistory { get; set; }
        public virtual DbSet<AppDatasets> AppDatasets { get; set; }
        #endregion

        #region Entity mapping
        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            modelBuilder.Entity<AppImage>().ToTable("images");
            modelBuilder.Entity<AppImageAttribute>().ToTable("imageattribute");
            modelBuilder.Entity<AppAttributeCategory>().ToTable("attributecategories");
            modelBuilder.Entity<AppAttribute>().ToTable("attributes");
            modelBuilder.Entity<AppUser>().ToTable("users");
            modelBuilder.Entity<AppTrainingHistory>().ToTable("traininghistory");
            modelBuilder.Entity<AppDatasets>().ToTable("datasets");
        }
        #endregion


        // see https://stackoverflow.com/a/53693858/6228814
        public static readonly ILoggerFactory DbCommandConsoleLoggerFactory = LoggerFactory.Create(builder =>
        {
            builder.AddConsole();
        });

        // see https://docs.microsoft.com/en-us/archive/msdn-magazine/2018/october/data-points-logging-sql-and-change-tracking-events-in-ef-core
        protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder)
        {
            optionsBuilder.UseLoggerFactory(DbCommandConsoleLoggerFactory).EnableSensitiveDataLogging();
        }
    }
}
