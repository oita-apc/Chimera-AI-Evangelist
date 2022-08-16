// Copyright (C) 2020 - 2022 APC, Inc.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using CorrectionWebApp.Models;
using CorrectionWebApp.Services;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Extensions.Configuration;
using webapp.Models;

namespace CorrectionWebApp.Controllers
{
    [Route("api/[controller]")]
    [ApiController]
    public class AnnotatorHelperController
    {
        protected AppService AppService { get; private set; }
        protected IConfiguration Configuration { get; private set; }
        public AnnotatorHelperController(AppService appService, IConfiguration configuration)
        {
            AppService = appService;
            Configuration = configuration;
        }        

        [HttpGet]
        [Route("GetAnnotations")]
        public virtual async Task<List<W3CWebAnnotationModel>> GetAnnotations(Guid imageId, int datasetId)
        {

            return await AppService.GetAnnotations(imageId, datasetId);
        }

        [HttpPost]
        [Route("DeleteAnnotation")]
        public virtual async Task<IActionResult> DeleteAnnotation (W3CWebAnnotationModel annotation)
        {
            var annotationId = GetAnnotationId(annotation);
            if (annotationId != Guid.Empty)
            {
                await AppService.DeleteAnnotation(annotationId);
            }

            return new OkResult();
        }

        [HttpPost]
        [Route("RegisterAnnotation")]
        public virtual async Task<ResponseModel> RegisterAnnotation(W3CWebAnnotationModel annotation, Guid imageId, Boolean isNew, int datasetId)
        {
            // validate annotation
            var validation = IsValidAnnotation(annotation);
            if (validation != null)
            {
                return validation;
            }
            var attributeId = await GetAttributeId(annotation);
            if (attributeId == null)
            {
                return new ResponseModel
                {
                    ResponseCode = 504,
                    ResponseMessage = "Label can not be found!"
                };
            }
            var annotationId = GetAnnotationId(annotation);
            if (annotationId == Guid.Empty)
            {
                return new ResponseModel
                {
                    ResponseCode = 506,
                    ResponseMessage = "Invalid annotationId!"
                };
            }
            
            var ret = await AppService.RegisterAnnotation(annotation.Target.Selector.Type, annotation.Target.Selector.Value, annotationId, attributeId.Value, imageId, datasetId);
            if (ret)
            {
                return new ResponseModel
                {
                    ResponseCode = 200
                };
            }
            else
            {
                return new ResponseModel
                {
                    ResponseCode = 510,
                    ResponseMessage = "Can not create annotation!"
                };
            }
        }

        private ResponseModel IsValidAnnotation(W3CWebAnnotationModel annotation)
        {
            if (annotation.Body == null || annotation.Body.Count == 0)
            {
                return new ResponseModel
                {
                    ResponseCode = 504,
                    ResponseMessage = "Label can not be found!"
                };
            }

            if (String.IsNullOrWhiteSpace(annotation.Id) ||
                annotation.Target == null ||
                annotation.Target.Selector == null)
            {
                return new ResponseModel
                {
                    ResponseCode = 505,
                    ResponseMessage = "Annotation format is invalid!"
                };
            }            

            return null;
        }

        private Guid GetAnnotationId(W3CWebAnnotationModel annotation)
        {
            if (!String.IsNullOrWhiteSpace(annotation.Id) && annotation.Id.Length > 2)
            {
                Guid annotationId;
                if (Guid.TryParse(annotation.Id.Substring(1), out annotationId))
                {
                    return annotationId;
                }
            }
            return Guid.Empty;
        }
        private async Task<int?> GetAttributeId(W3CWebAnnotationModel annotation)
        {
            if (annotation.Body.Count > 0)
            {
                var attributeName = annotation.Body[0].Value;
                if (!String.IsNullOrWhiteSpace(attributeName))
                {
                    var attribute = await AppService.getAttributeByName(attributeName);
                    if (attribute != null)
                    {
                        return attribute.Id;
                    }
                }
            }
            return null;
        }


    }
}
