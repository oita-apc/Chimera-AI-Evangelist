// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Logging;
using CorrectionWebApp.Models;
using Microsoft.AspNetCore.Authorization;
using System.Security.Claims;
using Microsoft.AspNetCore.Authentication.Cookies;
using Microsoft.AspNetCore.Authentication;
using CorrectionWebApp.Services;
using CorrectionWebApp.Helper;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Configuration;
using System.IO;
using System.Text.Json;

namespace CorrectionWebApp.Controllers
{
    public class HomeController : Controller
    {
        private readonly ILogger<HomeController> _logger;
        protected AppService AppService { get; private set; }
        protected IConfiguration Configuration { get; private set; }


        public HomeController(ILogger<HomeController> logger, AppService appService, IConfiguration configuration)
        {
            _logger = logger;
            AppService = appService;
            Configuration = configuration;
        }

        public async Task<IActionResult> LogOff()
        {
            try
            {
                var authenticationManager = Request.HttpContext;
                await authenticationManager.SignOutAsync(CookieAuthenticationDefaults.AuthenticationScheme);
                HttpContext.Session.Clear();
            }
            catch (Exception){}
            
            return this.RedirectToAction("Index");
        }

        [HttpGet]
        public IActionResult Index()
        {
            if (IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Dashboard");
            }
            ViewData["Title"] = "Chimera AI Evangelist";
            // ViewData["SubTitle"] = Configuration["ProductName"]; // "Object Detection";            
            return View();
        }

        [HttpPost]
        public async Task<IActionResult> IndexAsync(LoginViewModel model)
        {
            var passed = await AppService.login(model.Username, model.Password);
            if (passed)
            {
                // Login In.  
                await this.SignInUser(model.Username, true);
                return RedirectToAction("Dashboard");                
            }
            ViewData["Title"] = "Chimera AI Evangelist";
            // ViewData["SubTitle"] = "Image Classification";
            ViewData["error"] = "Incorrect login attempt";
            return View();
        }

        private async Task SignInUser(string username, bool isPersistent)
        {
            var claims = new List<Claim>();
            try
            {
                // Setting  
                claims.Add(new Claim(ClaimTypes.Name, username));
                var claimIdenties = new ClaimsIdentity(claims, CookieAuthenticationDefaults.AuthenticationScheme);
                var claimPrincipal = new ClaimsPrincipal(claimIdenties);
                var authenticationManager = Request.HttpContext;

                // Sign In.  
                await authenticationManager.SignInAsync(CookieAuthenticationDefaults.AuthenticationScheme, claimPrincipal, new AuthenticationProperties() { IsPersistent = isPersistent });
            }
            catch (Exception)
            {
                // Info  
                throw;
            }
        }

        [HttpPost]
        public async Task<IActionResult> TransferToAnnotator()
        {
            await AppService.TransferToAnnotator();
            return RedirectToAction("Dashboard");
        }

        public async Task<IActionResult> DashboardAsync()
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }

            for (int trial = 0; trial < 3; trial++)
            {
                var result = getTrainingStatus();
                await requestTrainingStatus();
                result.Wait(10 * 1000); // wait in max 10 secs
                if (result.IsCompletedSuccessfully)
                {
                    TrainStatusModel statusInfo = JsonSerializer.Deserialize<TrainStatusModel>(result.Result);
                    if ("idle".Equals(statusInfo.status))
                    {
                        var finishDate = new DateTime();
                        if (DateTime.TryParse(statusInfo.finish_date, out finishDate))
                        {
                            var jstZoneInfo = TimeZoneInfo.FindSystemTimeZoneById("Tokyo Standard Time");
                            finishDate = TimeZoneInfo.ConvertTimeFromUtc(finishDate, jstZoneInfo);
                        }
                        else
                        {
                            finishDate = DateTime.Now;
                        }
                        var modelType = statusInfo.model_type;
                        int datasetId;
                        if (int.TryParse(statusInfo.dataset_id, out datasetId))
                            await AppService.finishTraining(Helper.Const.TRAINING_STATUS_COMPLETED, finishDate, modelType, datasetId);
                    }
                    break;
                }
            }

            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Statistics Screen - ";

            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int currentDatasetId;
            if (!int.TryParse(datasetIdStr, out currentDatasetId))
            {
                currentDatasetId = 1;
            }
            */
            int currentDatasetId = await AppService.getActiveDatasetId();

            var model = await AppService.getSummaryData(currentDatasetId);
            model.IsAdministrator = await AppService.isAdministrator(this.User.Identity.Name);
            var datasets = await AppService.getDatasets();
            model.LastTrainingStatus = new List<TrainingHistoryModel>();
            foreach (var dataset in datasets)
            {
                var lastStatus = await AppService.getLastTrainingStatus(dataset.Id);
                if (lastStatus != null)
                {
                    lastStatus.DatasetName = String.Format("{0}. {1}", dataset.Id, dataset.Name);
                    lastStatus.IsActive = dataset.IsActive;
                    lastStatus.IsTrained = dataset.IsTrained;
                    lastStatus.DatasetId = dataset.Id;
                    model.LastTrainingStatus.Add(lastStatus);
                } else
                {
                    model.LastTrainingStatus.Add(new TrainingHistoryModel
                    {
                        Status = "NOT-TRAINING",
                        DatasetName = String.Format("{0}. {1}", dataset.Id, dataset.Name),
                        IsTrained = dataset.IsTrained,
                        IsActive = dataset.IsActive,
                        DatasetId = dataset.Id
                    });
                }
            }
            return View(model);
        }

        private async Task requestTrainingStatus()
        {
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {3}/trainstatusrequest -m {2}"
                , mqttclient, mqttbroker, "\"Hello mqtt\"", prefixTopic)
                .Bash(_logger);
        }

        private async Task<string> getTrainingStatus()
        {
            var mqttclient = Configuration["MqttClient:PathSub"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            var result = await String.Format("{0} -h {1} -t {2}/trainstatus -C 1"
                , mqttclient, mqttbroker, prefixTopic)
                .GetBash(_logger);

            return result;
        }

        [HttpPost]
        public async Task<IActionResult> ActivateDataset(int datasetId)
        {
            // send activate command to GPU
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {3}/activatemodel -m {2}"
                , mqttclient, mqttbroker, datasetId, prefixTopic)
                .Bash(_logger);
            await AppService.activateDataset(datasetId);
            return new OkResult();
        }

        [HttpPost]
        public async Task<IActionResult> StartTraining(int imageSize, string modelType, int datasetId)
        {
            await AppService.runTraining(this.User.Identity.Name, imageSize, modelType, datasetId);
            return new OkResult();
        }

        [HttpPost]
        public async Task<IActionResult> CompleteTraining(string modelType, int datasetId, string finishDate = "")        
        {
            var updateDate = new DateTime();
            if (DateTime.TryParse(finishDate, out updateDate))
            {
                var jstZoneInfo = TimeZoneInfo.FindSystemTimeZoneById("Tokyo Standard Time");
                updateDate = TimeZoneInfo.ConvertTimeFromUtc(updateDate, jstZoneInfo);
            }
            else
            {
                updateDate = DateTime.Now;
            }
   
            await AppService.finishTraining(Helper.Const.TRAINING_STATUS_COMPLETED, updateDate, modelType, datasetId);
            await ActivateDataset(datasetId);
            return new OkResult();
        }

        [HttpPost]
        public async Task<IActionResult> CancelTraining(string modelType, int datasetId)
        {
            await AppService.finishTraining(Helper.Const.TRAINING_STATUS_CANCELLED, DateTime.Now, modelType, datasetId);
            return Ok("");
        }

        public async Task<IActionResult> TrainingAsync()
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }
            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Training Screen - ";
            var labels = await AppService.getTrainingLabels();
            var model = new TrainingViewModel
            {
                Labels = labels.Where(x => !String.IsNullOrEmpty(x.Name)).OrderBy(x => x.OrderNo).Select(x => x.Name).ToList()
            };
            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int datasetId;
            if (!int.TryParse(datasetIdStr, out datasetId))
            {
                datasetId = 1;
            }
            */
            int datasetId = await AppService.getActiveDatasetId();
            model.NumberOfData = await AppService.getNumberOfData(datasetId);
            model.DatasetType = await AppService.getCurrentDatasetType(datasetId);
            return View(model);
        }

        public async Task<IActionResult> LabelsAsync()
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }
            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Labels Screen - ";
            var labels = await AppService.getLabels();
            var model = new Models.LabelsModel();
            model.Labels = labels.Select(x => new LabelModel
            {
                Id = x.Id,
                Name = x.Name,
                OrderNo = x.OrderNo
            }).OrderBy(x => x.OrderNo).ToList();            
            return View(model);
        }

        [HttpPost]
        public async Task<IActionResult> LabelsAsync(ICollection<Models.LabelModel> model)
        {
            await AppService.SaveLabels(model);
            return RedirectToAction("Labels");
        }

        public async Task<IActionResult> DatasetsAsync()
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }
            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Datasets Screen - ";
            var datasets = await AppService.getDatasets();
            var model = new Models.DatasetsModel();
            model.Datasets = datasets.OrderBy(x => x.Id).ToList();
            return View(model);
        }

        [HttpPost]
        public async Task<IActionResult> UpdateDatasetAsync(Dataset dataset)
        {
            await AppService.updateDataset(dataset);
            return RedirectToAction("Datasets");
        }

        [HttpPost]
        public async Task<IActionResult> DeleteDatasetAsync(Dataset dataset)
        {
            // remove also corresponding model on GPU
            await removeModelInGPU(dataset.Id);
            await AppService.deleteDataset(dataset);
            return RedirectToAction("Datasets");
        }

        private async Task removeModelInGPU(int datasetId)
        {
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {3}/removemodel -m {2}"
                , mqttclient, mqttbroker, datasetId, prefixTopic)
                .Bash(_logger);
        }

        [HttpPost]
        public async Task<IActionResult> CreateDatasetAsync(Dataset dataset)
        {
            await AppService.createDataset(dataset);
            return RedirectToAction("Datasets");
        }


        [HttpGet]
        public  IActionResult DoClear()
        {
            HttpContext.Session.SetObjectAsJson("search_params", new SearchImage());
            return Ok("");
        } 

        public async Task<IActionResult> ListAsync(SearchImage filter, int? imgno)
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }


            var pageNo = filter.PageNo;
            if (filter.Submitted == null)
            {
                var _filter = HttpContext.Session.GetObjectFromJson<SearchImage>("search_params");
                if (_filter != null)
                {
                    filter = _filter;
                }
                else
                {
                    filter = new SearchImage();
                }
            } else
            {
                HttpContext.Session.SetObjectAsJson("search_params", filter);
            }
            if (filter.PageSize == 0)
                filter.PageSize = filter.ViewMode == 0 ? Helper.Const.PAGE_SIZE_LIST : Helper.Const.PAGE_SIZE_THUMBNAIL;

            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int datasetId;
            if (!int.TryParse(datasetIdStr, out datasetId))
            {
                datasetId = 1;
            }
            */

            int datasetId = await AppService.getActiveDatasetId();

            #region adjust page size
            if (filter.ViewMode == 0 && !ImageListModel.ListPageSize.Exists(x => x == filter.PageSize)){
                filter.PageSize = ImageListModel.ListPageSize.OrderBy(x => Math.Abs(x - filter.PageSize)).First();
            } else if (filter.ViewMode == 1 && !ImageListModel.ThumbnailPageSize.Exists(x => x == filter.PageSize))
            {
                filter.PageSize = ImageListModel.ThumbnailPageSize.OrderBy(x => Math.Abs(x - filter.PageSize)).First();
            }
            #endregion

            var rawData = await AppService.getRawData(filter, datasetId);
            if (imgno != null)
            {                
                filter.PageNo = AppService.getPageNoByImageNo(rawData, imgno.Value, filter.PageSize, filter);
            } else
            {
                filter.PageNo = pageNo;
            }
            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Image List Screen - ";
            var model = await AppService.getImageListData(rawData, filter, filter.PageSize, datasetId);
            model.AttributeList = await AppService.getAttributeList(filter);
            model.SearchImage = filter;
            model.IsAdministrator = await AppService.isAdministrator(this.User.Identity.Name);
            return View(model);
        }

        public async Task<IActionResult> Detail(Guid id)
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }
            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int datasetId;
            if (!int.TryParse(datasetIdStr, out datasetId))
            {
                datasetId = 1;
            }
            */
            int datasetId = await AppService.getActiveDatasetId();

            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Image Detail Screen - ";
            var model = await AppService.getDetailData(id, datasetId);
            if (model == null)
                return NotFound();
            var filter = new SearchImage();
            var _filter = HttpContext.Session.GetObjectFromJson<SearchImage>("search_params");
            if (_filter != null)
            {
                filter = _filter;
            }
            
            var rawData = await AppService.getRawData(filter, datasetId);
            model.AttributeList = await AppService.getAttributeList(filter);
            model.PrevImageId = AppService.getPrevImage(model.ImageNo, model.CreatedDate, rawData, filter);
            model.NextImageId = AppService.getNextImage(model.ImageNo, model.CreatedDate, rawData, filter);
            model.IsAdministrator = await AppService.isAdministrator(this.User.Identity.Name);
            model.DatasetType = await AppService.getCurrentDatasetType(datasetId);
            model.DatasetId = datasetId;
            
            return View("Detail", model);
        }

        [HttpPost]
        public async Task<ActionResult> Detail(DetailModel detail)
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }

            if (detail == null ||  detail.Id == Guid.Empty)
                return NotFound();

            if (detail.Attribute > 0)
            {
                detail.Status = ImageStatus.Confirmed;
            } else
            {
                detail.Status = ImageStatus.Unconfirmed;
            }
            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int datasetId;
            if (!int.TryParse(datasetIdStr, out datasetId))
            {
                datasetId = 1;
            }
            */
            int datasetId = await AppService.getActiveDatasetId();
            var currentData = await AppService.getDetailData(detail.Id, datasetId);
            if (currentData == null)
                return NotFound();
            
            var ret = await AppService.Save(detail);
            detail.AttributeList = new AttributeListModel();
            if (ret)
            {
                    return RedirectToAction("Detail", new { id = detail.Id });
            }

            // set current data            
            detail.AttributeAll = currentData.AttributeAll;
            detail.CreatedDate = currentData.CreatedDate;
            detail.ImageNo = currentData.ImageNo;
            detail.ImageFileName = currentData.ImageFileName;

            return View(detail);
        }

        [HttpPost]
        public async Task<IActionResult> ChangeLabel(Guid id, int attribute, ImageStatus status, int datasetId)
        {
            var detail = new DetailModel
            {
                Id = id,
                Attribute = attribute,
                Status = status,
                DatasetId = datasetId
            };
            
            var result = await Detail(detail);

            return new OkResult();
        }

        [HttpGet]
        public IActionResult Documents()
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }

            ViewData["Title"] = "Chimera AI Evangelist";
            ViewData["SubTitle"] = " - Documents Screen - ";
            return View();
        }


        private async Task PublishDeleteImage(String filename)
        {
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {3}/deleteimage/{2} -m {2}"
                , mqttclient, mqttbroker, filename, prefixTopic)
                .Bash(_logger);
        }

        private async Task PublishImage(String filename, String labelId)
        {
            var imageStorageDirectoryPath = Configuration["ImageStorage:Directory"];
            var imageFilepath = Path.Combine(imageStorageDirectoryPath, filename);
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {4}/pushimage/{2} -f {3}"
                , mqttclient, mqttbroker, filename, imageFilepath, prefixTopic)
                .Bash(_logger);
        }

        private async Task PublishBulk(List<Guid> checkbox, ImageStatus action, Boolean forceDelete = false)
        {
            var images = await AppService.getImages(checkbox);
            var attributCategories = await AppService.getAttributeList(new SearchImage());
            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int datasetId;
            if (!int.TryParse(datasetIdStr, out datasetId))
            {
                datasetId = 1;
            }
            */
            int datasetId = await AppService.getActiveDatasetId();
            var imageAttributes = await AppService.getImageAttributeList(datasetId);
            var attributes = attributCategories.AttributeList.Where(x => x.Id == 1);
            foreach (var image in images)
            {
                if (forceDelete)
                {
                    if (image.Status == (int)ImageStatus.Confirmed)
                        await PublishDeleteImage(image.ImageFileName);
                }
                else if (image.Status != (int)action)
                {
                    if (image.Status == (int)ImageStatus.Confirmed)
                    {
                        // previouly was confirmed, lets remove the image on gpu server
                        await PublishDeleteImage(image.ImageFileName);
                    }
                    else if (action == ImageStatus.Confirmed)
                    {
                        var imageAttribute = imageAttributes.FirstOrDefault(x => x.ImageId == image.Id);

                        if (imageAttribute != null && attributes.Count() > 0)
                        {
                            var attribute = attributes.First().Attributes.FirstOrDefault(x => x.Id == imageAttribute.AttributeId);
                            if (attribute != null)
                            {
                                // push the image to the gpu server
                                await PublishImage(image.ImageFileName, attribute.Id.ToString());
                            }
                        }
                    }
                }
            };
        }


        [HttpPost]
        public async Task<ActionResult> BulkApprove(List<Guid> checkbox)
        {
            await PublishBulk(checkbox, ImageStatus.Confirmed);
            var imgno = await AppService.BulkAction(checkbox, ImageStatus.Confirmed);
            return RedirectToAction("List", new { imgno = imgno });
        }

        [HttpPost]
        public async Task<ActionResult> BulkReject(List<Guid> checkbox)
        {
            await PublishBulk(checkbox, ImageStatus.Rejected);
            var imgno = await AppService.BulkAction(checkbox, ImageStatus.Rejected);
            return RedirectToAction("List", new { imgno = imgno });
        }

        [HttpPost]
        public async Task<ActionResult> BulkUnconfirm(List<Guid> checkbox)
        {
            await PublishBulk(checkbox, ImageStatus.Unconfirmed);
            var imgno = await AppService.BulkAction(checkbox, ImageStatus.Unconfirmed);
            return RedirectToAction("List", new { imgno = imgno });
        }

        
        [HttpPost]
        public async Task<ActionResult> BulkNotEntered(List<Guid> checkbox)
        {
            await PublishBulk(checkbox, ImageStatus.NotEntered);
            var imgno = await AppService.BulkAction(checkbox, ImageStatus.NotEntered);
            return RedirectToAction("List", new { imgno = imgno });
        }

        
        [HttpPost]
        public async Task<ActionResult> BulkDelete(List<Guid> checkbox)
        {
            await PublishBulk(checkbox, ImageStatus.Rejected, true); // to simulate delete
            if (checkbox.Count == 0)
                return RedirectToAction("List");
            var img = await AppService.getImageById(checkbox[0]);
            if (img == null)
                return RedirectToAction("List");
            foreach (var imageId in checkbox)
            {
                await AppService.Remove(imageId);
            }
            return RedirectToAction("List", new { imgno = img.ImageNo });
        }


        [HttpPost]
        public async Task<ActionResult> Remove(DetailModel detail)
        {
            if (!IsAuthenticated())
            {
                // Login screen
                return this.RedirectToAction("Index");
            }

            if (detail == null || detail.Id == Guid.Empty)
                return NotFound();

            /*
            var datasetIdStr = Request.Cookies["datasetId"];
            int datasetId;
            if (!int.TryParse(datasetIdStr, out datasetId))
            {
                datasetId = 1;
            }
            */
            int datasetId = await AppService.getActiveDatasetId();
            var currentData = await AppService.getDetailData(detail.Id, datasetId);
            if (currentData == null)
                return NotFound();

            if (currentData.Status == ImageStatus.Confirmed)
                await PublishDeleteImage(currentData.ImageFileName);

            await AppService.Remove(detail.Id);

            var filter = new SearchImage();
            var _filter = HttpContext.Session.GetObjectFromJson<SearchImage>("search_params");
            if (_filter != null)
            {
                filter = _filter;
            }
            
            var rawData = await AppService.getRawData(filter, datasetId);
            var nextImageId = AppService.getNextImage(currentData.ImageNo, currentData.CreatedDate, rawData, filter);
            if (nextImageId != null)
            {
                // redirect to next image
                return RedirectToAction("Detail", new { id = nextImageId.Value });
            }
            else
            {
                // redirect to image list
                return RedirectToAction("List", new { imgno = detail.ImageNo });
            }            
            
        }

        private bool IsAuthenticated()
        {
            if (this.User != null && this.User.Identity != null && this.User.Identity.IsAuthenticated)
            {
                return true;
            }
            return false;
        }

        [ResponseCache(Duration = 0, Location = ResponseCacheLocation.None, NoStore = true)]
        public IActionResult Error()
        {
            return View(new ErrorViewModel { RequestId = Activity.Current?.Id ?? HttpContext.TraceIdentifier });
        }
    }
}
