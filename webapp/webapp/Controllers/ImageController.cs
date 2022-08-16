using System;
using System.IO;
using System.Threading.Tasks;
using CorrectionWebApp.Services;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.Logging;

namespace CorrectionWebApp.Controllers
{
    public class ImageController : Controller
    {
        private readonly ILogger<ImageController> _logger;
        protected AppService AppService { get; private set; }
        protected IConfiguration Configuration { get; private set; }
        private readonly IWebHostEnvironment _env;

        public ImageController(ILogger<ImageController> logger,
            AppService appService, IWebHostEnvironment env,
            IConfiguration configuration)
        {
            _logger = logger;
            AppService = appService;
            _env = env;
            Configuration = configuration;
        }

        [HttpGet]
        [AllowAnonymous]
        public  IActionResult Index(string f, bool t = false) // filename
        {
            var filepathNoImage = Path.Combine(_env.WebRootPath, "img", "blank-image.png");
            var fileBytesNoImage = System.IO.File.ReadAllBytes(filepathNoImage);
            try
            {
                var imageStorageDirectoryPath = Configuration["ImageStorage:Directory"];
                if (t)
                {
                    imageStorageDirectoryPath = Configuration["ImageStorage:DirectoryThumbnail"];
                }
                var imageFilepath = Path.Combine(imageStorageDirectoryPath, f);
                if (System.IO.File.Exists(imageFilepath))
                {
                    var fileBytes = System.IO.File.ReadAllBytes(imageFilepath);
                    return File(fileBytes, "image/*", f);
                }
                return File(fileBytesNoImage, "image/*", f);
            }
            catch (Exception)
            {                
                return File(fileBytesNoImage, "image/*", f);
            }
        }
    }
}
