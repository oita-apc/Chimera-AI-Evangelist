// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Mvc;
using CorrectionWebApp.swagger;
using CorrectionWebApp.Models;
using Microsoft.AspNetCore.Http;
using CorrectionWebApp.Services;
using System.IO;
using Microsoft.Extensions.Configuration;
using System.Globalization;
using CorrectionWebApp.Helper;
using System.Linq;
using System.Collections.Generic;
using Microsoft.Extensions.Logging;

using System.Text.Json;
using System.Xml;
using webapp.Models;
using System.Xml.Serialization;

namespace CorrectionWebApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class ImageUploaderController
    {
        protected AppService AppService { get; private set; }
        protected IConfiguration Configuration { get; private set; }
        protected ILogger Logger { get; private set; }
        public ImageUploaderController(AppService appService, IConfiguration configuration, ILogger<ImageUploaderController> logger)
        {
            AppService = appService;
            Configuration = configuration;
            Logger = logger;
        }

        [HttpPost]
        [Route("Login")]
        public virtual async Task<UserModel> Login(UserModel model)
        {
            var passed = await AppService.login(model.Id, model.Password);
            if (passed)
            {
                return await AppService.getUser(model.Id);
            }
            return new UserModel();
        }

        [HttpGet]
        [Route("GetLabel")]
        public virtual async Task<string> GetLabel(int id)
        {
            var label = await AppService.getLabel(id);
            var dic = new Dictionary <string, string>{
                {"label", label}
            };

            return JsonSerializer.Serialize(dic);
        }

        [HttpGet]
        [Route("GetActiveModel")]
        public virtual async Task<string> GetActiveModel()
        {
            int currentDatasetId = await AppService.getActiveDatasetId();
            var lastStatus = await AppService.getLastTrainingStatus(currentDatasetId);
            var dataset =  await AppService.getDatasetById(currentDatasetId);
            Dictionary<string, string> dic = new Dictionary<string, string>();
            if(lastStatus == null)
            {
                dic = new Dictionary <string, string>{
                    {"dataset", dataset.Name},
                    {"model", "NOT-TRAINING"}
                };
            }
            else{
                dic = new Dictionary <string, string>{
                    {"dataset", dataset.Name},
                    {"model", lastStatus.ModelType}
                };
            }
            Console.WriteLine("ModelType : " + dic["model"]);
            Console.WriteLine("DatasetName : " + dic["dataset"]);
            return JsonSerializer.Serialize(dic);
        }


        [HttpPost]
        [SwaggerUploadFile(Parameter = "file")]
        [Route("UploadHeartbeat")]
        [Consumes("multipart/form-data")]
        public virtual ResponseModel UploadHeartbeat([FromForm] IFormFile file)
        {
            AppService.UploadHeartbeat(file);
            Console.WriteLine("UploadHeartbeat");
            return new ResponseModel
            {
                ResponseCode = 200
            };
        }

        [HttpPost]
        [SwaggerUploadFile(Parameter = "file")]
        [Route("UploadImage")]
        [Consumes("multipart/form-data")]
        public virtual async Task<ResponseModel> UploadImage(
            [FromQuery] string userName,
            [FromQuery] string attributes,
            [FromQuery] string timestamp,
            [FromForm] IFormFile file
        )
        {
            var imageStorageDirectory = Configuration["ImageStorage:Directory"];
            var IsDirExists = Directory.Exists(imageStorageDirectory);
            if (!IsDirExists)
            {
                return new ResponseModel
                {
                    ResponseCode = 501,
                    ResponseMessage = "サーバーエラーが発生しました。管理者に連絡してください。"
                };
            }
            if (file.Length == 0)
            {
                return new ResponseModel
                {
                    ResponseCode = 502,
                    ResponseMessage = "エラーが発生しました。もう一度試してくださいもしくは管理者に連絡してください。"
                };
            }

            // Check GPU server down
            var result = getTrainingStatus();
            await requestTrainingStatus();
            result.Wait(3 * 1000); // wait in max 3 secs
            if (!result.IsCompletedSuccessfully)
            {
                Console.WriteLine("gpu server down");
                return new ResponseModel
                {
                    ResponseCode = 503,
                    ResponseMessage = "エラーが発生しました。もう一度試してくださいもしくは管理者に連絡してください。"
                };
            }
            
            var createdDate = DateTime.Now;
            if (!string.IsNullOrWhiteSpace(timestamp))
            {
                long timestamp_long;
                if (long.TryParse(timestamp, out timestamp_long))
                {
                    // Unix timestamp is seconds past epoch
                    var dtDateTime = new DateTime(1970, 1, 1, 0, 0, 0, 0, System.DateTimeKind.Utc);
                    dtDateTime = dtDateTime.AddMilliseconds(timestamp_long).ToLocalTime();
                    createdDate = dtDateTime;
                }                
            }

            var filename = await AppService.RegisterImageAsync(userName, file, int.Parse(attributes), createdDate);
            if (!String.IsNullOrEmpty(filename))
            {
                await PublishImage(filename);
                return new ResponseModel
                {
                    ResponseCode = 200
                };
            } else
            {
                return new ResponseModel
                {
                    ResponseCode = 503,
                    ResponseMessage = "エラーが発生しました。もう一度試してくださいもしくは管理者に連絡してください。"
                };
            }
        }
        
        private async Task requestTrainingStatus()
        {
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {3}/trainstatusrequest -m {2}"
                , mqttclient, mqttbroker, "\"Hello mqtt\"", prefixTopic)
                .Bash(Logger);
        }

        private async Task<string> getTrainingStatus()
        {
            var mqttclient = Configuration["MqttClient:PathSub"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            var result = await String.Format("{0} -h {1} -t {2}/trainstatus -C 1"
                , mqttclient, mqttbroker, prefixTopic)
                .GetBash(Logger);

            return result;
        }


        private async Task PublishImage(String filename)
        {
            var imageStorageDirectoryPath = Configuration["ImageStorage:Directory"];
            var imageFilepath = Path.Combine(imageStorageDirectoryPath, filename);
            var mqttclient = Configuration["MqttClient:Path"];
            var mqttbroker = Configuration["MqttClient:Broker"];
            var prefixTopic = Configuration["MqttClient:PrefixTopic"];
            await String.Format("{0} -h {1} -t {4}/pushimage/{2} -f {3}"
                , mqttclient, mqttbroker, filename, imageFilepath, prefixTopic)
                .Bash(Logger);
        }

        [HttpGet]
        [Route("JsonDatasetResnetV2")]
        public virtual async Task<ResnetV2DatasetModel> getResnetV2Dataset(int datasetId)
        {
            // get all  images
            var filter = new SearchImage();
            var rawData = await AppService.getRawData(filter, datasetId);

            // get categories
            var attributes = await AppService.getLabels();
            var imageAttributes = await AppService.getImageAttributeList(datasetId);

            List<Entities.AppImage> trainDataset = new List<Entities.AppImage>();
            List<Entities.AppImage> valDataset = new List<Entities.AppImage>();

            // clean up data
            foreach (var label in attributes)
            {
                var labelData = rawData.Where(x => x.Width > 0 && x.Height > 0
                    && imageAttributes.Exists(y => y.ImageId == x.Id && y.AnnotationId == null && y.AttributeId == label.Id)).ToList();

                if (labelData.Count() == 0)
                    continue;
                
                // get train and val dataset
                // we will divide training:80%  and validation:20%
                Random rnd = new Random();
                var randomList = labelData.Select(x => new { value = x, order = rnd.Next() })
                    .OrderBy(x => x.order).Select(x => x.value).ToList();
                
                var trainCount = (int)(randomList.Count * 0.8);
                List<Entities.AppImage> labelTrainDataset = randomList.Take(trainCount).ToList();
                List<Entities.AppImage> labelvalDataset = randomList.Skip(trainCount).ToList();
                trainDataset.AddRange(labelTrainDataset);
                valDataset.AddRange(labelvalDataset);
            }

            var beingUsedLabelIds = imageAttributes
                .Where(x => rawData.Exists(y => y.Id == x.ImageId))
                .Select(x => x.AttributeId).Distinct().ToList();

            var train = trainDataset.Select(x => new ResnetV2Image
            {
                FileName = x.ImageFileName,
                ClassId = imageAttributes.First(y => y.ImageId == x.Id).AttributeId.ToString()
            }).ToList();


            return new ResnetV2DatasetModel
            {
                Train = new ResnetV2ImageList {
                    Images = trainDataset.Select(x => new ResnetV2Image
                    {
                        FileName = x.ImageFileName,
                        ClassId = imageAttributes.First(y => y.ImageId == x.Id).AttributeId.ToString()
                    }).ToList()
                },
                Val = new ResnetV2ImageList
                {
                    Images = valDataset.Select(x => new ResnetV2Image
                    {
                        FileName = x.ImageFileName,
                        ClassId = imageAttributes.First(y => y.ImageId == x.Id).AttributeId.ToString()
                    }).ToList()
                },
                Labels = attributes.Where(x => beingUsedLabelIds.Exists(y => x.Id == y))
                .Select(x => new ResnetV2Category(x.Id, x.Name)).ToList()
            };
        }

        [HttpGet]
        [Route("JsonDataset")]
        public virtual async Task<CocoAllDatasetModel> getCocoDataset(int datasetId)
        {
            Dataset datasetInfo = await AppService.getDatasetById(datasetId);
            if (datasetInfo == null)
                return new CocoAllDatasetModel();
            // get all  images
            var filter = new SearchImage();
            var rawData = await AppService.getRawData(filter, datasetId);

            // get categories
            var attributes = await AppService.getLabels();
            var imageAttributes = await AppService.getImageAttributeList(datasetId);

            List<Entities.AppImage> trainDataset = new List<Entities.AppImage>();
            List<Entities.AppImage> valDataset = new List<Entities.AppImage>();

            // clean up data
            foreach (var label in attributes)
            {
                var labelData = rawData.Where(x => x.Width > 0 && x.Height > 0
                    && imageAttributes.Exists(y => y.ImageId == x.Id && y.AnnotationId != null && y.AttributeId == label.Id)).ToList();

                if (labelData.Count() == 0)
                    continue;

                // get train and val dataset
                // we will divide training:80%  and validation:20%
                Random rnd = new Random();
                var randomList = labelData.Select(x => new { value = x, order = rnd.Next() })
                .OrderBy(x => x.order).Select(x => x.value).ToList();
                var trainCount = (int)(randomList.Count * 0.8);
                List<Entities.AppImage> labelTrainDataset = randomList.Take(trainCount).ToList();
                List<Entities.AppImage> labelvalDataset = randomList.Skip(trainCount).ToList();
                trainDataset.AddRange(labelTrainDataset);
                valDataset.AddRange(labelvalDataset);
            }
            
            var beingUsedLabelIds = imageAttributes
                .Where(x => rawData.Exists(y => y.Id == x.ImageId))
                .Select(x => x.AttributeId).Distinct().ToList();

            return new CocoAllDatasetModel
            {
                Train = createCocoDataset(trainDataset, attributes, imageAttributes.Where(x => trainDataset.Exists(y => y.Id == x.ImageId)).ToList(), datasetInfo.Type),
                Val = createCocoDataset(valDataset, attributes, imageAttributes.Where(x => valDataset.Exists(y => y.Id == x.ImageId)).ToList(), datasetInfo.Type),
                //Labels = attributes.Select(x => new CategoryCoco(x.Id, x.Name)).ToList()
                Labels = attributes.Where(x => beingUsedLabelIds.Exists(y => x.Id == y))
                .Select(x => new CategoryCoco(x.Id, x.Name)).ToList()
            };
        }

        private CocoDatasetModel createCocoDataset(List<Entities.AppImage> images, List<Models.Attribute> attributes, List<Entities.AppImageAttribute> imageAttributes, string datasetType)
        {
            var dataset = new CocoDatasetModel();

            dataset.Categories = imageAttributes
                .Where(x => images.Exists(y => y.Id == x.ImageId))
                .Select(x => x.AttributeId).Distinct()
                .Select(x => new CategoryCoco(x, x.ToString())).ToList();
            dataset.Images = images.Select(x => new ImageCoco
            {
                Id = x.ImageNo,
                FileName = x.ImageFileName,
                Height = x.Height,
                Width = x.Width
            }).ToList();

            dataset.Annotations = new List<AnnotationCoco>();
            var index = 1;
            foreach (var imageAttribute in imageAttributes)
            {
                if (String.IsNullOrWhiteSpace(imageAttribute.Data))
                {
                    Logger.LogError("Annotation data is empty --> imageAttributeId:" + imageAttribute.Id + " annotationId:" + imageAttribute.AnnotationId);
                    continue;
                }

                var image = images.FirstOrDefault(x => x.Id == imageAttribute.ImageId);
                if (image == null)
                {
                    Logger.LogError("Image is null --> imageId:" + imageAttribute.ImageId);
                    continue;
                }
                if (imageAttribute.AnnotationType == "FragmentSelector" && datasetType == Helper.Const.OBJECT_DETECTION)
                {
                    var xywhStr = imageAttribute.Data.Substring("xywh=pixel:".Length);
                    var coco = createCocoByFragmentSelector(xywhStr);
                    if (coco != null)
                    {
                        coco.Id = index++;
                        coco.ImageId = image.ImageNo;
                        coco.CategoryId = imageAttribute.AttributeId;
                        dataset.Annotations.Add(coco);
                    }                                 
                } else if (imageAttribute.AnnotationType == "SvgSelector" && datasetType == Helper.Const.INSTANCE_SEGMENTATION) {
                    var svgStr = imageAttribute.Data;
                    var serializer = new XmlSerializer(typeof(SvgSelector));
                    var coco = createCocoBySvgSelector(svgStr, serializer);
                    if (coco != null)
                    {
                        coco.Id = index++;
                        coco.ImageId = image.ImageNo;
                        coco.CategoryId = imageAttribute.AttributeId;
                        dataset.Annotations.Add(coco);
                    }
                }
                else
                {
                    Logger.LogError("AnnotationType is not supported -->  annotationType:" + imageAttribute.AnnotationType);
                }
            }            
            return dataset;
        }

        private AnnotationCoco createCocoBySvgSelector(String svgStr, XmlSerializer serializer)
        {                        
            try
            {
                using (StringReader sr = new StringReader(svgStr))
                {                    
                    var svgObj = (SvgSelector)serializer.Deserialize(sr);
                    if (svgObj != null && svgObj.polygon != null && !string.IsNullOrWhiteSpace(svgObj.polygon.points))
                    {
                        var pointsStr = svgObj.polygon.points.Split(" ");                        
                        if (pointsStr.Length > 2)
                        {
                            var points = new List<int>();
                            var pointXYstr = pointsStr[0].Split(",");
                            if (pointXYstr.Count() == 2)
                            {
                                var minX = Double.Parse(pointXYstr[0]);
                                var minY = Double.Parse(pointXYstr[1]);
                                var maxX = minX;
                                var maxY = minY;
                                points.Add((int)minX);
                                points.Add((int)minY);
                                for (int i = 1; i < pointsStr.Length - 1; i++)
                                {
                                    pointXYstr = pointsStr[i].Split(",");
                                    if (pointXYstr.Count() == 2)
                                    {
                                        var x = Double.Parse(pointXYstr[0]);
                                        var y = Double.Parse(pointXYstr[1]);
                                        points.Add((int)x);
                                        points.Add((int)y);
                                        minX = Math.Min(x, minX);
                                        maxX = Math.Max(x, maxX);
                                        minY = Math.Min(y, minY);
                                        maxY = Math.Max(y, maxY);
                                    }

                                }
                                if (points.Count > 2)
                                {
                                    var w = (int)(maxX - minX);
                                    var h = (int)(maxY - minY);
                                    var coco = new AnnotationCoco
                                    {
                                        Area = (int)(w * h),
                                        Segmentation = new List<List<int>> { points },
                                        Bbox = new List<int> {
                                        (int)minX, (int)minY,       // x, y -> top left
                                        w,          // width
                                        h           // height
                                    }
                                    };
                                    return coco;
                                }
                            }
                        }                        
                    }
                }
                return null;
            } catch (Exception ex)
            {
                Logger.LogError("failed to create coco from SvgSelector " + ex.ToString(), ex);
            }
            return null;
        }

        private AnnotationCoco createCocoByFragmentSelector(String xywhStr)
        {
            var xywh = xywhStr.Split(",");
            if (xywh.Length != 4)
            {
                Logger.LogError("xywh array lenght is not 4  --> length:" + xywh.Length);
                return null;
            }
            double _x, _y, _w, _h;
            if (Double.TryParse(xywh[0], out _x)
                && Double.TryParse(xywh[1], out _y)
                && Double.TryParse(xywh[2], out _w)
                && Double.TryParse(xywh[3], out _h)
                )
            {
                Logger.LogInformation("x:" + _x + " y:" + _y + " w:" + _w + " h:" + _h);
                var x = (int)Math.Round(_x);
                var y = (int)Math.Round(_y);
                var w = (int)Math.Round(_w);
                var h = (int)Math.Round(_h);
                var coco = new AnnotationCoco
                {
                    Area = w * h,
                    Segmentation = new List<List<int>> { new List<int> {
                                x, y,           // top left
                                x + w, y,       // top right
                                x + w, y + h,   // bottom right
                                x, y + h        // bottom left
                            } },
                    Bbox = new List<int> {
                                x, y,       // x, y -> top left
                                w,          // width
                                h           // height
                            }
                };
                return coco;
            }
            Logger.LogError("Failed to parse xywh --> " + xywhStr);
            return null;
        }

        private Entities.AppImage getImage(Entities.AppImageAttribute imageAttribute, List<Entities.AppImage> images)
        {
            return images.FirstOrDefault(x => x.Id == imageAttribute.ImageId);
        }

    }
}
