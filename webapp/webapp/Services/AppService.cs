// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using CorrectionWebApp.Entities;
using Microsoft.EntityFrameworkCore;
using System.Linq;
using System.Threading.Tasks;
using System.IO;
using Microsoft.AspNetCore.Http;
using CorrectionWebApp.Models;
using CorrectionWebApp.Helper;
using System.Collections.Generic;
using Microsoft.Extensions.Configuration;
using System.Drawing;
using System.Drawing.Imaging;
using System.Drawing.Drawing2D;
using System.Globalization;
//using ExifLib;
using System.Net.Http;
using System.Diagnostics;
using webapp.Models;

namespace CorrectionWebApp.Services
{
    public class AppService
    {
        protected AppDbContext AppDb { get; private set; }
        protected IConfiguration Configuration { get; private set; }
        public AppService(AppDbContext appDb, IConfiguration configuration)
        {
            AppDb = appDb;
            Configuration = configuration;
        }

        //static Image FixedSize(Image imgPhoto, int Width, int Height)
        //{
        //    int sourceWidth = imgPhoto.Width;
        //    int sourceHeight = imgPhoto.Height;
        //    int sourceX = 0;
        //    int sourceY = 0;
        //    int destX = 0;
        //    int destY = 0;

        //    float nPercent = 0;
        //    float nPercentW = 0;
        //    float nPercentH = 0;

        //    nPercentW = ((float)Width / (float)sourceWidth);
        //    nPercentH = ((float)Height / (float)sourceHeight);
        //    if (nPercentH < nPercentW)
        //    {
        //        nPercent = nPercentH;
        //        destX = System.Convert.ToInt16((Width -
        //                      (sourceWidth * nPercent)) / 2);
        //    }
        //    else
        //    {
        //        nPercent = nPercentW;
        //        destY = System.Convert.ToInt16((Height -
        //                      (sourceHeight * nPercent)) / 2);
        //    }

        //    int destWidth = (int)(sourceWidth * nPercent);
        //    int destHeight = (int)(sourceHeight * nPercent);

        //    Bitmap bmPhoto = new Bitmap(Width, Height,
        //                      PixelFormat.Format24bppRgb);
        //    if (imgPhoto.HorizontalResolution != 0 && imgPhoto.VerticalResolution != 0)
        //        bmPhoto.SetResolution(imgPhoto.HorizontalResolution, imgPhoto.VerticalResolution);

        //    Graphics grPhoto = Graphics.FromImage(bmPhoto);
        //    grPhoto.Clear(Color.Black);
        //    grPhoto.InterpolationMode =
        //            InterpolationMode.HighQualityBicubic;

        //    grPhoto.DrawImage(imgPhoto,
        //        new Rectangle(destX, destY, destWidth, destHeight),
        //        new Rectangle(sourceX, sourceY, sourceWidth, sourceHeight),
        //        GraphicsUnit.Pixel);

        //    grPhoto.Dispose();
        //    return bmPhoto;
        //}

        public async Task SetAnnotation(int annotatorId, int annotation)
        {
            var image = await AppDb.AppImages.FirstOrDefaultAsync(x => x.AnnotatorId == annotatorId);
            if (image != null)
            {
                image.TransferredToAnnotation = (int)AnnotationTransferredEnum.Transferred;
                image.IsAnotated = annotation > 0;
                AppDb.AppImages.Update(image);
                await AppDb.SaveChangesAsync();
            }
        }

        public async Task<AppImage> getImageByAnnotatorId(int annotatorId)
        {
            return await AppDb.AppImages.FirstOrDefaultAsync(x => x.AnnotatorId == annotatorId);
        }

        public async Task SetTransferred(Guid imageId, int annotatorId)
        {
            var image = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == imageId);
            if (image != null)
            {
                image.TransferredToAnnotation = (int)AnnotationTransferredEnum.Transferred;
                image.AnnotatorId = annotatorId;
                AppDb.AppImages.Update(image);
                await AppDb.SaveChangesAsync();
            }
        }

        public async Task TransferToAnnotator()
        {

            var sqlStr = string.Format("UPDATE images SET TransferredToAnnotation = {0} WHERE Status = {1} AND TransferredToAnnotation = {2}",
                (int) AnnotationTransferredEnum.Transferring, (int)ImageStatus.Confirmed, (int)AnnotationTransferredEnum.NotYet);
            await AppDb.Database.ExecuteSqlRawAsync(sqlStr);
            try
            {
                using (var httpClient = new HttpClient())
                {
                    httpClient.Timeout = TimeSpan.FromSeconds(10);
                    var url = Configuration["AnnotatorApi:ScanDataset"];
                    using (var response = await httpClient.GetAsync(url))
                    {
                        string apiResponse = await response.Content.ReadAsStringAsync();
                        //reservationList = JsonConvert.DeserializeObject<List<Reservation>>(apiResponse);
                        Console.WriteLine("api:" + url + " apiResponse:" + apiResponse);
                    }
                }
            } catch(Exception ex)
            {
                Console.WriteLine("call annotator api failed + " + ex.ToString());
            }

        }

        public async Task<List<Models.Attribute>> getLabels()
        {
            return await AppDb.AppAttributes.Where(x => x.CategoryId == 1).OrderBy(x => x.OrderNo)
                .Select(x => new Models.Attribute
                {
                    Id = x.Id,
                    Name = x.Name,
                    OrderNo = x.OrderNo
                })
                .ToListAsync();
        }

        public async Task<Dataset> getDatasetById(int datasetId)
        {
            return await AppDb.AppDatasets.Where(x => x.Id == datasetId).Select(x => new Models.Dataset
                {
                Id = x.Id,
                    Name = x.Name,
                    Type = x.Type
                }).FirstOrDefaultAsync();
        }

        public async Task<List<Models.Dataset>> getDatasets()
        {
            return await AppDb.AppDatasets.OrderBy(x => x.Id)
                .Select(x => new Models.Dataset
                {
                    Id = x.Id,
                    Name = x.Name,
                    Type = x.Type,
                    IsActive = x.IsActive,
                    IsTrained = x.IsTrained
                })
                .ToListAsync();
        }

        public async Task<List<W3CWebAnnotationModel>> GetAnnotations(Guid imageId, int datasetId)
        {
            var image = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == imageId);
            if (image == null)
                return new List<W3CWebAnnotationModel>();
            var appImageAttributes = await AppDb.AppImageAttribute.Where(x => x.ImageId == imageId && x.AnnotationId != null && x.DatasetId == datasetId).AsNoTracking().ToListAsync();
            if (appImageAttributes == null || appImageAttributes.Count == 0)
                return new List<W3CWebAnnotationModel>();
            var appAttributes = await AppDb.AppAttributes.ToListAsync();
            if (appAttributes == null || appAttributes.Count == 0)
                return new List<W3CWebAnnotationModel>();
            return appImageAttributes.Select(x => new W3CWebAnnotationModel
            {
                Id = "#" + x.AnnotationId.ToString(),
                Body = new List<Body>
                {
                    new Body
                    {
                        Value = appAttributes.FirstOrDefault(y => y.Id == x.AttributeId) != null ? appAttributes.FirstOrDefault(y => y.Id == x.AttributeId).Name : ""
                    }
                },
                Target = new Target
                {
                    Source = image.ImageFileName, // TODO: construct url of the image
                    Selector = new Selector
                    {
                        Type = x.AnnotationType,
                        Value = x.Data
                    }
                }
            }).ToList();
        }

        public async Task activateDataset(int datasetId)
        {
            var datasets = await AppDb.AppDatasets.ToListAsync();
            foreach(var dataset in datasets)
            if (dataset != null)
            {
                dataset.IsActive = dataset.Id == datasetId ? true : false;
                AppDb.AppDatasets.Update(dataset);
            }
            await AppDb.SaveChangesAsync();
        }

        public async Task<string> getLabel(int id)
        {
            var label = await AppDb.AppAttributes.FirstOrDefaultAsync(x => x.Id == id);
            if(label != null) 
            {
                return label.Name;
            }
            return string.Empty;
        }


        public async Task<List<Models.Attribute>> getTrainingLabels()
        {
            var allLabels = await AppDb.AppAttributes.Where(x => x.CategoryId == 1).OrderBy(x => x.OrderNo)
                .Select(x => new Models.Attribute
                {
                    Id = x.Id,
                    Name = x.Name,
                    OrderNo = x.OrderNo
                })
                .ToListAsync();
            
            var imageLabels = await AppDb.AppImageAttribute.Select(x => x.AttributeId).Distinct().ToListAsync();
            var beingUsedLabels = allLabels.Where(x => imageLabels.Exists(y => y == x.Id)).ToList();
            return beingUsedLabels.ToList();
        }

        public async Task<int> getNumberOfData(int datasetId)
        {
            var images = await AppDb.AppImages.ToListAsync();
            var imageAttributes = await AppDb.AppImageAttribute.Where(x => x.DatasetId == datasetId).ToListAsync();
            return images.Where(x => imageAttributes.Exists(y => y.ImageId == x.Id)).Count();
        }

        public async Task<Boolean> RegisterAnnotation(String annotationType, String annotationData, Guid annotationId, int attributeId, Guid imageId, int datasetId)
        {
            var appImageAttribute = await AppDb.AppImageAttribute.FirstOrDefaultAsync(x => x.AnnotationId == annotationId);
            if (appImageAttribute != null)
            {
                appImageAttribute.AttributeId = attributeId;
                appImageAttribute.AnnotationType = annotationType;
                appImageAttribute.Data = annotationData;
                appImageAttribute.DatasetId = datasetId;
                AppDb.AppImageAttribute.Update(appImageAttribute);
            } else
            {
                // insert to attributes
                AppDb.AppImageAttribute.Add(new AppImageAttribute
                {
                    AttributeId = attributeId,
                    ImageId = imageId,
                    AnnotationId = annotationId,
                    AnnotationType = annotationType,
                    Data = annotationData,
                    DatasetId = datasetId
                });
            }
            return await AppDb.SaveChangesAsync() == 1;
        }

        public async Task<bool> updateDataset(Dataset dataset)
        {
            var appDataset = await AppDb.AppDatasets.FirstOrDefaultAsync(x => x.Id == dataset.Id);
            if (appDataset != null)
            {
                appDataset.Name = dataset.Name;
                AppDb.AppDatasets.Update(appDataset);
                return await AppDb.SaveChangesAsync() == 1;
            }
            return false;
        }

        public async Task<bool> createDataset(Dataset dataset)
        {
            var lastDataset = await AppDb.AppDatasets.OrderByDescending(x => x.Id).FirstOrDefaultAsync();
            var appDataset = new AppDatasets();
            appDataset.Id = lastDataset.Id + 1;
            appDataset.Name = dataset.Name;
            appDataset.Type = dataset.Type;
            AppDb.AppDatasets.Add(appDataset);
            return await AppDb.SaveChangesAsync() == 1;
        }

        public async Task<Boolean> DeleteAnnotation(Guid annotationId)
        {
            var appImageAttribute = await AppDb.AppImageAttribute.FirstOrDefaultAsync(x => x.AnnotationId == annotationId);
            if (appImageAttribute != null)
            {
                AppDb.AppImageAttribute.Remove(appImageAttribute);
                return await AppDb.SaveChangesAsync() == 1;
            }
            return true;
        }

        public async Task<bool> deleteDataset(Dataset dataset)
        {
            var appDataset = await AppDb.AppDatasets.FirstOrDefaultAsync(x => x.Id == dataset.Id);
            if (appDataset != null)
            {
                AppDb.AppDatasets.Remove(appDataset);
                return await AppDb.SaveChangesAsync() == 1;
            }
            return false;
        }

        public async Task<List<ImageInfo>> GetAllImagesForAnnotationAsync()
        {
            return await AppDb.AppImages.Where(x => x.Status == (int) ImageStatus.Confirmed && x.TransferredToAnnotation != (int)AnnotationTransferredEnum.Transferred)
                .Select(x => new ImageInfo
                {
                    Id = x.Id,
                    FileName = x.ImageFileName,
                    ImageNo = x.ImageNo
                }).AsNoTracking().ToListAsync();
        }

        public async Task<AppAttribute> getAttributeByName(string attributeName)
        {
            return await AppDb.AppAttributes.FirstOrDefaultAsync(x => x.Name == attributeName);

        }

        public async Task SaveLabels(ICollection<Models.LabelModel> labels)
        {
            foreach(var label in labels)
            {
                var entity = await AppDb.AppAttributes.FirstOrDefaultAsync(x => x.Id == label.Id);
                entity.Name = !String.IsNullOrWhiteSpace(label.Name) ? label.Name : "";
                AppDb.Update(entity);
            }
            await AppDb.SaveChangesAsync();
        }

        public async Task<string> getCurrentDatasetType(int datasetId)
        {
            var datasets = await AppDb.AppDatasets.AsNoTracking().ToListAsync();
            var dataset = datasets.FirstOrDefault(x => x.Id == datasetId);
            if (dataset != null)
            {
                return dataset.Type;
            }
            return datasets[0].Type;
        }

        public async Task<int> getActiveDatasetId()
        {
            var datasets = await AppDb.AppDatasets.AsNoTracking().ToListAsync();
            var dataset = datasets.FirstOrDefault(x => x.IsActive);
            if (dataset != null)
            {
                return dataset.Id;
            }
            return datasets[0].Id;
        }


        public async Task<UserModel> getUser(string username)
        {
            return await AppDb.AppUsers.Where(x => x.Id == username).Select(x => new UserModel
            {
                Id = x.Id,
                Name = x.Name,
                Role = x.Role
            }).FirstOrDefaultAsync();
        }

        public async Task<bool> login(string username, string password)
        {
            var user = await AppDb.AppUsers.FirstOrDefaultAsync(x => x.Id == username && x.Password == password);
            return user != null;
        }

        public async Task<String> RegisterImageAsync(string userName, IFormFile file, int combinedAttributes, DateTime createdDatetime)
        {
            var imageId = Guid.NewGuid();
            // insert to image table
            var entity = new AppImage
            {
                Id = imageId,
                CreatedDate = createdDatetime,
                Status = 0,
                UserName = userName,
                ImageFileName = ""
            };
            AppDb.AppImages.Add(entity);
            await AppDb.SaveChangesAsync();
            // select again
            entity = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == imageId);

            var imageNoStr = entity.ImageNo;
            var fileExtension = Path.GetExtension(file.FileName);
            if (String.IsNullOrEmpty(fileExtension)) {
                fileExtension = ".jpg";
            }
            var imageFileName = $"IMG_{imageNoStr.ToString("D4")}{fileExtension}";

            // save the image to storage directory
            var imageStorageDirectoryPath = Configuration["ImageStorage:Directory"];
            var imageFilepath = Path.Combine(imageStorageDirectoryPath, imageFileName);

            var width = 0;
            var height = 0;

            ExifData exifData = null;
            try
            {
                using (var imageStream = file.OpenReadStream())
                {
                    // get exif data
                    exifData = Helper.Helper.GetExifData(imageStream);
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine("ExifData exception occured :" + ex.ToString());
            }

            // just save raw file (this is much efficient
            using (Stream fileStream = new FileStream(imageFilepath, FileMode.Create))
            {
                await file.CopyToAsync(fileStream);
            }

            /* this is to heavy on aws lightsail instance (it can cause whole system down)
            try
            {
                using (var imageStream = file.OpenReadStream())
                {
                    // create the file image
                    var image = Image.FromStream(imageStream);
                    if (exifData != null && exifData.orientation != 0)
                    {
                        image.FixImageOrientation(exifData);
                    }
                    width = image.Width;
                    height = image.Height;
                    image.Save(imageFilepath);

                    //create thumbnail
                    var thumb = FixedSize(image, 128 * 3, 72 * 3);
                    var imageStorageThumbnailDirectoryPath = Configuration["ImageStorage:DirectoryThumbnail"];
                    var imageThumbnailFilepath = Path.Combine(imageStorageThumbnailDirectoryPath, imageFileName);
                    thumb.Save(imageThumbnailFilepath);
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine("FixImageOrientation or Save file exception occured :" + ex.ToString());
            }
            */
            try
            {
                using (var p = new Process())
                {
                    var imageStorageThumbnailDirectoryPath = Configuration["ImageStorage:DirectoryThumbnail"];
                    var imageThumbnailFilepath = Path.Combine(imageStorageThumbnailDirectoryPath, imageFileName);

                    // e.g. command --> $convert -thumbnail 364 ~/QA/datasets/airplane/IMG_2742.JPG thumbnail.JPG && ./magick identify ~/QA/datasets/airplane/IMG_2742.JPG
                    p.StartInfo.WorkingDirectory = Configuration["ThumbnailCreator:WorkingDirectory"]; // "/home/ubuntu/QA/magick";
                    p.StartInfo.FileName = Configuration["ThumbnailCreator:FileName"]; //"/home/ubuntu/QA/magick/imageprocess.sh";
                    p.StartInfo.UseShellExecute = false;
                    p.StartInfo.RedirectStandardOutput = true;                    
                    var thumbnailWidth = 384;
                    var thumbnailHeight = 216;
                    var thumbnailSize = thumbnailWidth + "x" + thumbnailHeight;
                    // arguments 1:thumbnailSize 2:imageFilepath  3:imageThumbnailFilepath
                    p.StartInfo.Arguments = thumbnailSize + " " + imageFilepath + " " + imageThumbnailFilepath;
                    p.Start();
                    string cmdOutputStr = p.StandardOutput.ReadToEnd();
                    var arrInfo = cmdOutputStr.Split(" ");
                    if (arrInfo.Length >= 3)
                    {
                        var sizeStr = arrInfo[2];
                        var arrSize = sizeStr.Split("x");
                        if (arrSize.Length == 2)
                        {
                            width = int.Parse(arrSize[0]);
                            height = int.Parse(arrSize[1]);
                        }
                    }
                    p.WaitForExit();
                }
            } catch(Exception ex)
            {
                Console.WriteLine("FixImageOrientation or Save file exception occured :" + ex.ToString());
            }

            // update the table
            entity.ImageFileName = imageFileName;
            entity.Width = width;
            entity.Height = height;
            if (exifData != null)
            {
                if (exifData.Latitude != null && exifData.Longitude != null)
                {
                    entity.Latitude = exifData.Latitude;
                    entity.Longitude = exifData.Longitude;
                }
                if (exifData.DateTimeOriginal != null)
                {
                    entity.CreatedDate = exifData.DateTimeOriginal.Value;
                }
            }
            AppDb.AppImages.Update(entity);

            var attributes = await AppDb.AppAttributes.AsNoTracking().ToListAsync();
            foreach(var appAttribute in attributes)
            {
                if ((appAttribute.Id & combinedAttributes) == appAttribute.Id)
                {
                    // insert to attributes
                    AppDb.AppImageAttribute.Add(new AppImageAttribute
                    {
                        AttributeId = appAttribute.Id,
                        ImageId = imageId
                    });
                }
            }

            await AppDb.SaveChangesAsync();

            return imageFileName;
        }

        public bool UploadHeartbeat(IFormFile file)
        {
            var imageFileName = $"IMG_HEARTBEAT{Path.GetExtension(file.FileName)}";

            // save the image to storage directory
            var imageStorageDirectoryPath = Configuration["ImageStorage:Directory"];
            var imageFilepath = Path.Combine(imageStorageDirectoryPath, imageFileName);
            //var width = 0;
            //var height = 0;

            ExifData exifData = null;
            try
            {
                using (var imageStream = file.OpenReadStream())
                {
                    // get exif data
                    exifData = Helper.Helper.GetExifData(imageStream);
                }

            }
            catch (Exception)
            {
                //Console.WriteLine("exception occured :" + ex.ToString());
            }

            try
            {
                using (var imageStream = file.OpenReadStream())
                {
                    //// create the file image
                    //var image = Image.FromStream(imageStream);
                    //if (exifData != null && exifData.orientation != 0)
                    //{
                    //    image.FixImageOrientation(exifData);
                    //}
                    //width = image.Width;
                    //height = image.Height;
                    //image.Save(imageFilepath);

                    ////create thumbnail
                    //var thumb = FixedSize(image, 128 * 3, 72 * 3);
                    //var imageStorageThumbnailDirectoryPath = Configuration["ImageStorage:DirectoryThumbnail"];
                    //var imageThumbnailFilepath = Path.Combine(imageStorageThumbnailDirectoryPath, imageFileName);
                    //thumb.Save(imageThumbnailFilepath);
                }

            }
            catch (Exception ex)
            {
                Console.WriteLine("exception occured :" + ex.ToString());
            }
            

            return true;
        }

        public int getPageNoByImageNo(List<AppImage> rawData, int imageNo, int pageSize, SearchImage filter)
        {            
            var img = rawData.FirstOrDefault(x => x.ImageNo == imageNo);
            int imageCount = 0;
            if (filter.OrderByField == (int)OrderByEnum.ImageNo || img == null)
            {
                imageCount = rawData.Where(x => x.ImageNo >= imageNo).Count();
            } else {
                imageCount = rawData.Where(x =>
                    (x.CreatedDate == img.CreatedDate && x.ImageNo >= imageNo) ||
                    (x.CreatedDate > img.CreatedDate)
                ).Count();
            }

            if (imageCount == 0)
                return 0;

            return imageCount / pageSize + (imageCount % pageSize == 0 ? 0 : 1);

        }

        public async Task<AppImage> getImageById(Guid guid)
        {
            return await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == guid);
        }

        public async Task<List<AppImage>> getImages(List<Guid> guids)
        {
            var rawData = await AppDb.AppImages.AsNoTracking().ToListAsync();
            return rawData.Where(x => guids.Exists(y => y == x.Id)).ToList();
        }

        public async Task<int> BulkAction(List<Guid> guids, ImageStatus action)
        {

            int imgno = 1;
            var needToSave = false;
            foreach(var guid in guids)
            {
                var image = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == guid);
                if (image != null)
                {
                    image.Status = (int)action;
                    AppDb.AppImages.Update(image);
                    imgno = image.ImageNo;
                    needToSave = true;
                }
            }
            if (needToSave)
                await AppDb.SaveChangesAsync();
            return imgno;


        }

        public async Task<bool> isAdministrator(string userId)
        {
            var user = await AppDb.AppUsers.FirstOrDefaultAsync(x => x.Id == userId);
            if (user != null)
            {
                return user.Role == "administrator";
            }
            return false;
        }

        public Guid? getPrevImage(int currentImageNo, DateTime currentCreatedDatetime, List<AppImage> rawData, SearchImage filter)
        {
            AppImage prevImage = null;
            if (filter.OrderByField == (int) OrderByEnum.ImageNo)
                prevImage = rawData.Where(x => x.ImageNo > currentImageNo).OrderBy(x => x.ImageNo).FirstOrDefault();
            else
                prevImage = rawData.Where(x => x.CreatedDate > currentCreatedDatetime).OrderBy(x => x.CreatedDate).ThenBy(x => x.ImageNo).FirstOrDefault();
            if (prevImage != null)
                return prevImage.Id;
            else
                return null;
        }

        public Guid? getNextImage(int currentImageNo, DateTime currentCreatedDatetime, List<AppImage> rawData, SearchImage filter)
        {
            AppImage nextImage = null;
            if (filter.OrderByField == (int) OrderByEnum.ImageNo)
                nextImage = rawData.Where(x => x.ImageNo < currentImageNo).OrderByDescending(x => x.ImageNo).FirstOrDefault();
            else
                nextImage = rawData.Where(x => x.CreatedDate < currentCreatedDatetime).OrderByDescending(x => x.CreatedDate).ThenByDescending(x => x.ImageNo).FirstOrDefault();
            if (nextImage != null)
                return nextImage.Id;
            else
                return null;
        }

        public async Task<bool> Save(DetailModel detail)
        {
            var imageEntity = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == detail.Id);
            if (imageEntity != null)
            {
                imageEntity.Comment = detail.Comment;
                imageEntity.Status = (int)detail.Status;
                AppDb.AppImages.Update(imageEntity);

                // update attribute
                var oldAttributes = await AppDb.AppImageAttribute.Where(x => x.ImageId == detail.Id && x.DatasetId == detail.DatasetId).ToListAsync();
                AppDb.AppImageAttribute.RemoveRange(oldAttributes);
                if (detail.Attributes != null)
                {
                    foreach (var attributeId in detail.Attributes)
                    {
                        // insert to attributes
                        AppDb.AppImageAttribute.Add(new AppImageAttribute
                        {
                            AttributeId = attributeId,
                            ImageId = detail.Id,
                            DatasetId = detail.DatasetId
                        });
                    }
                }
                if (detail.Attribute > 0)
                {
                    // insert to attributes
                    AppDb.AppImageAttribute.Add(new AppImageAttribute
                    {
                        AttributeId = detail.Attribute,
                        ImageId = detail.Id,
                        DatasetId = detail.DatasetId
                    });
                }

                return await AppDb.SaveChangesAsync() > 0;
            }

            return false;
        }

        public async Task<bool> Remove(Guid imageId)
        {
            var imageEntity = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == imageId);
            if (imageEntity != null)
            {
                var imageFileName = imageEntity.ImageFileName;                
                var imageStorageDirectoryPath = Configuration["ImageStorage:Directory"];
                var imageFilepath = Path.Combine(imageStorageDirectoryPath, imageFileName);
                var imageStorageThumbnailDirectoryPath = Configuration["ImageStorage:DirectoryThumbnail"];
                var imageThumbnailFilepath = Path.Combine(imageStorageThumbnailDirectoryPath, imageFileName);
                try
                {
                    File.Delete(imageFilepath);
                    File.Delete(imageThumbnailFilepath);
                } catch(Exception ex)
                {
                    Console.WriteLine("failed to remove imageId:" + imageId + " ex:" + ex.ToString());
                }

                AppDb.AppImages.Remove(imageEntity);

                // remove attribute
                var oldAttributes = await AppDb.AppImageAttribute.Where(x => x.ImageId == imageId).ToListAsync();
                AppDb.AppImageAttribute.RemoveRange(oldAttributes);

                return await AppDb.SaveChangesAsync() > 0;
            }

            return false;
        }        

        public async Task<DetailModel> getDetailData(Guid imageId, int datasetId)
        {
            var detailImage = await AppDb.AppImages.FirstOrDefaultAsync(x => x.Id == imageId);
            if (detailImage != null)
            {
                var user = await AppDb.AppUsers.FirstOrDefaultAsync(x => x.Id == detailImage.UserName);
                var attributes = await AppDb.AppAttributes.OrderBy(x => x.OrderNo).AsNoTracking().ToListAsync();
                var imageAttributes = await AppDb.AppImageAttribute.Where(x => x.ImageId == imageId && x.DatasetId == datasetId).ToListAsync();
                var result = new DetailModel
                {
                    Id = detailImage.Id,
                    Comment = detailImage.Comment,
                    CreatedDate = detailImage.CreatedDate,
                    ImageFileName = detailImage.ImageFileName,
                    ImageNo = detailImage.ImageNo,
                    Status = (ImageStatus)detailImage.Status,
                    Height = detailImage.Height,
                    Width = detailImage.Width,
                    UserName = user != null ? user.Id : detailImage.UserName,
                    Latitude = detailImage.Latitude,
                    Longitude = detailImage.Longitude,
                    AnnotatorId = detailImage.AnnotatorId
                };
                result.AttributeAll = attributes.OrderBy(x => x.OrderNo).Select(x => new AttributeItem { Id = x.Id, Name = x.Name }).ToList();
                if (imageAttributes != null)
                {
                    result.Attributes = imageAttributes.Select(x => x.AttributeId).ToList();
                    if (result.Attributes.Count > 0)
                        result.Attribute = result.Attributes.First();
                }
                else
                    result.Attributes = new List<int>();


                return result;
            }

            return null;
        }


        public async Task<List<AppImage>> getRawData(SearchImage filter, int datasetId)
        {
            var rawData = await AppDb.AppImages.AsNoTracking().ToListAsync();
            #region where clause
            if (!string.IsNullOrWhiteSpace(filter.SearchComment))
            {
                // filter by comment
                rawData = rawData.Where(x => x.Comment != null && x.Comment.Contains(filter.SearchComment)).ToList();
            }
            if (filter.SearchAttributes != null)
            {
                var searchAttributes = filter.SearchAttributes.Where(x => x.IndexOf("|") > 0).Select(x => x.Substring(x.IndexOf("|") + 1)).ToList();
                if (searchAttributes.Count() > 0)
                {
                    rawData = (from a in rawData
                               join b in AppDb.AppImageAttribute on a.Id equals b.ImageId
                               where searchAttributes.Exists(x => int.Parse(x) == b.AttributeId)
                               && b.DatasetId == datasetId
                               select a).Distinct().ToList();
                }
            }
            if (!String.IsNullOrEmpty(filter.SearchAttribute) && filter.SearchAttribute != "0")
            {
                rawData = (from a in rawData
                           join b in AppDb.AppImageAttribute on a.Id equals b.ImageId
                           where b.AttributeId == int.Parse(filter.SearchAttribute)
                           && b.DatasetId == datasetId
                           select a).Distinct().ToList();
            }
            if (filter.SearchStatus != null)
            {
                rawData = rawData.Where(x => x.Status == filter.SearchStatus.Value).ToList();
            }
            if (!string.IsNullOrWhiteSpace(filter.SearchPhotographer))
            {
                //rawData = rawData.Where(x => x.UserName != null && x.UserName.Contains(filter.SearchPhotographer)).ToList();
                var users = await AppDb.AppUsers.AsNoTracking().ToListAsync();
                rawData = (from a in rawData
                          join b in users on a.UserName equals b.Id
                          where b.Id.Contains(filter.SearchPhotographer) || b.Name.Contains(filter.SearchPhotographer)
                          select a).ToList();
            }

            if (!string.IsNullOrWhiteSpace(filter.DateStartStr))
            {
                DateTime dateTime;
                if (DateTime.TryParseExact(filter.DateStartStr, "yyyy/MM/dd", null, DateTimeStyles.None, out dateTime))
                {
                    rawData = rawData.Where(x => x.CreatedDate >= dateTime).ToList();
                }
            }

            if (!string.IsNullOrWhiteSpace(filter.DateEndStr))
            {
                DateTime dateTime;
                if (DateTime.TryParseExact(filter.DateEndStr, "yyyy/MM/dd", null, DateTimeStyles.None, out dateTime))
                {
                    dateTime = dateTime.AddDays(1).AddSeconds(-1);
                    rawData = rawData.Where(x => x.CreatedDate <= dateTime).ToList();
                }
            }

            if (filter.ImageNoFrom != null)
            {
                rawData = rawData.Where(x => x.ImageNo >= filter.ImageNoFrom.Value).ToList();
            }
            if (filter.ImageNoEnd != null)
            {
                rawData = rawData.Where(x => x.ImageNo <= filter.ImageNoEnd.Value).ToList();
            }

            if (filter.AlreadyTransferred)
            {
                rawData = rawData.Where(x => x.AnnotatorId != null).ToList();
            }

            if (filter.IsAnnotated != null)
            {
                rawData = rawData.Where(x => x.IsAnotated == filter.IsAnnotated.Value).ToList();
            }

            #endregion

            return rawData;
        }

        public async Task<ImageListModel> getImageListData(List<AppImage> rawData, SearchImage filter, int pageSize, int datasetId)
        {
            var dataImageAttributes = await AppDb.AppImageAttribute.Where(x => x.DatasetId == datasetId).AsNoTracking().ToListAsync();
            var users = await AppDb.AppUsers.AsNoTracking().ToListAsync();            

            var result = new ImageListModel();
            List<AppImage> dataImages;
            if (filter.OrderByField == (int) OrderByEnum.ImageNo)
                dataImages = rawData.OrderByDescending(x => x.ImageNo).Skip((filter.PageNo - 1) * pageSize).Take(pageSize).ToList();
            else
                dataImages = rawData.OrderByDescending(x => x.CreatedDate).ThenByDescending(x => x.ImageNo).Skip((filter.PageNo - 1) * pageSize).Take(pageSize).ToList();

            var attributes = await AppDb.AppAttributes.OrderBy(x => x.OrderNo).AsNoTracking().ToListAsync();
            result.TotalCount = rawData.Count();

            result.Images = new List<ImageItem>();
            foreach(var image in dataImages)
            {
                var user = users.FirstOrDefault(x => x.Id == image.UserName);
                var imageItem = new ImageItem
                {
                    Comment = image.Comment,
                    Id = image.Id,
                    CreatedDate = image.CreatedDate,
                    ImageFileName = image.ImageFileName,
                    ImageNo = image.ImageNo,
                    Status = (ImageStatus)image.Status,
                    UserName = user != null ? user.Id : image.UserName,
                    Width = image.Width,
                    Height = image.Height,
                    AnnotatorId = image.AnnotatorId
                };

                var _imageAttributeQry = from a in dataImages
                                        join b in dataImageAttributes on a.Id equals b.ImageId
                                        join c in attributes on b.AttributeId equals c.Id
                                        where a.Id == image.Id
                                        select new AttributeItem
                                        {
                                            Id = c.Id,
                                            Name = c.Name
                                        };

                imageItem.Attributes = _imageAttributeQry.ToList();
                result.Images.Add(imageItem);
            }

            return result;
        }

        public async Task<SummaryModel> getSummaryData(int datasetId)
        {
            var result = new SummaryModel();
            var dataImages = await AppDb.AppImages.AsNoTracking().ToListAsync();
            var dataImageAttributes = await AppDb.AppImageAttribute.Where(x => x.DatasetId == datasetId).AsNoTracking().ToListAsync();
            var attributes = await AppDb.AppAttributes.Where(x => !String.IsNullOrWhiteSpace(x.Name)).AsNoTracking().ToListAsync();
            var categories = await AppDb.AppAttributeCategory.OrderBy(x => x.OrderNo).ToListAsync();

            var statuses = new List<ImageStatus> { ImageStatus.Confirmed, ImageStatus.Rejected, ImageStatus.Unconfirmed, ImageStatus.NotEntered };
            result.Statuses = statuses;

            // summary status
            result.StatusSummaries = new List<StatusSummaryDto>();            
/* 
*            foreach(var status in statuses)
*            {
*                result.StatusSummaries.Add(new StatusSummaryDto
*                {
*                    Status = status,
*                    Count = dataImages.Count(x => x.Status == (int)status)
*                }); 
*            }
*/
            result.TotalImage = dataImages.Count();

            // summary attribute
            result.AttributeSummaries = new List<AttributeCategorySummaryDto>();
            foreach (var category in categories)
            {
                var _category = new AttributeCategorySummaryDto
                {
                    Id = category.Id,
                    Name = category.Name,
                    OrderNo = category.OrderNo,
                    Attributes = new List<AttributeStatusSummaryDto>()
                };
                foreach (var attribute in attributes.Where(x => x.CategoryId == category.Id).OrderBy(x => x.OrderNo))
                {
                    var attributeSummary = new AttributeStatusSummaryDto
                    {
                        AttributeId = attribute.Id,
                        AttributeName = attribute.Name,
                        TotalImage = dataImageAttributes.Count(x => x.AttributeId == attribute.Id),
                        StatusSummaries = new List<StatusSummaryDto>()
                    };
/*                     
*                    foreach (var status in statuses)
*                    {
*                        var imageAttributeQry = from a in dataImages
*                                                join b in dataImageAttributes on a.Id equals b.ImageId
*                                                where b.AttributeId == attribute.Id
*                                                select new { image = a, imageAttribute = b };
*                        attributeSummary.StatusSummaries.Add(new StatusSummaryDto
*                        {
*                            Status = status,
*                            Count = imageAttributeQry.Count(x => x.image.Status == (int)status)
*                        });
*                    }
*/
                    _category.Attributes.Add(attributeSummary);
                }
                result.AttributeSummaries.Add(_category);
            }

            // annotator transferring status
            result.TransferringSummary = new AnnotatorTransferringSummary
            {
                TotalAlreadyTransferred = dataImages.Count(x => x.TransferredToAnnotation == (int) AnnotationTransferredEnum.Transferred),
                TotalAlreadyConfirmedNotTransffered = dataImages.Count(x => x.Status == (int) ImageStatus.Confirmed && x.TransferredToAnnotation == (int) AnnotationTransferredEnum.NotYet),
                TotalStillTransferring = dataImages.Count(x => x.TransferredToAnnotation == (int) AnnotationTransferredEnum.Transferring)
            };

            return result;
        }

        public async Task<AttributeListModel> getAttributeList(SearchImage filter)
        {
            var categories = await AppDb.AppAttributeCategory.AsNoTracking().ToListAsync();
            var attributes = await AppDb.AppAttributes.Where(x => !String.IsNullOrWhiteSpace(x.Name)).AsNoTracking().ToListAsync();
            var list = new AttributeListModel();
            list.AttributeList = new List<AttributeCategory>();
            foreach(var category in categories.OrderBy(x => x.OrderNo))
            {
                var attributeCategory = new AttributeCategory
                {
                    Id = category.Id,
                    Name = category.Name,
                    OrderNo = category.OrderNo,
                    Checked = filter.SearchAttributes != null && filter.SearchAttributes.Exists(x =>
                        x == category.Id.ToString()
                    )
                };
                attributeCategory.Attributes = new List<Models.Attribute>();
                foreach(var attribute in attributes.Where(x => x.CategoryId == category.Id).OrderBy(x => x.OrderNo))
                {
                    //var attributeId = id.substring(id.indexOf("|") + 1);
                    attributeCategory.Attributes.Add(new Models.Attribute
                    {
                        Id = attribute.Id,
                        Name = attribute.Name,
                        OrderNo = attribute.OrderNo,
                        Checked = filter.SearchAttributes != null && filter.SearchAttributes.Exists(x => x.IndexOf("|") > 0
                            && x.Substring(x.IndexOf("|") + 1) == attribute.Id.ToString())
                    });
                }
                list.AttributeList.Add(attributeCategory);
            }
            return list;
        }

        public async Task<List<AppImageAttribute>> getImageAttributeList(int datasetId)
        {
            return await AppDb.AppImageAttribute.Where(x => x.DatasetId == datasetId).AsNoTracking().ToListAsync();
        }

        public async Task runTraining(string userName, int imageSize, string modelType, int datasetId)
        {
            var entity = new AppTrainingHistory
            {
                UserName = userName,
                Status = Helper.Const.TRAINING_STATUS_RUNNING,
                ImageSize = imageSize,
                StartDate = DateTime.Now,
                FinishDate = null,
                ModelType = modelType,
                DatasetId = datasetId
            };
            AppDb.AppTrainingHistory.Add(entity);
            await AppDb.SaveChangesAsync();
        }

        public async Task finishTraining(string status, DateTime finishDate, string modelType, int datasetId)
        {
            var entity = await getTrainingHistory(datasetId);
            if(entity != null && entity.Status.Equals(Helper.Const.TRAINING_STATUS_RUNNING)) {
                entity.Status = status;
                entity.FinishDate = finishDate;
                entity.ModelType = modelType;
                entity.DatasetId = datasetId;
                AppDb.AppTrainingHistory.Update(entity);
                await AppDb.SaveChangesAsync();
            }
            if (status == Helper.Const.TRAINING_STATUS_COMPLETED)
            {
                var dataset = await AppDb.AppDatasets.FirstOrDefaultAsync(x => x.Id == datasetId);
                if (dataset != null)
                {
                    dataset.IsTrained = true;
                    AppDb.AppDatasets.Update(dataset);
                    await AppDb.SaveChangesAsync();
                }
            }
        }

        public async Task<TrainingHistoryModel> getLastTrainingStatus(int datasetId)
        {
            var entity = await getTrainingHistory(datasetId);
            if(entity != null)
            {
                var result = new TrainingHistoryModel{
                    UserName = entity.UserName,
                    Status = entity.Status,
                    ImageSize = entity.ImageSize != 0 ? entity.ImageSize.ToString() : "",
                    StartDate = entity.StartDate,
                    FinishDate = entity.FinishDate,
                    ModelType = entity.ModelType
                };

                return result;
            }
            
            return null;
        }

        public async Task<AppTrainingHistory> getTrainingHistory(int datasetId)
        {
            return await AppDb.AppTrainingHistory.Where(x =>
                (x.Status == Helper.Const.TRAINING_STATUS_RUNNING
                || x.Status == Helper.Const.TRAINING_STATUS_COMPLETED)
                && x.DatasetId == datasetId
            ).OrderByDescending(x => x.StartDate).FirstOrDefaultAsync();
        }
    }
}
