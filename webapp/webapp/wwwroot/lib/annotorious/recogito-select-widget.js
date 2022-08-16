var ColorSelectorWidget = function(args) {
    // console.log("ColorSelectorWidget args -> " + JSON.stringify(args));
    var purpose = _.get(args, 'purpose', 'tagging');

    // 1. Find a current color setting in the annotation, if any
    var currentSelectBody = args.annotation ? 
        args.annotation.bodies.find((b) => {
        return b.purpose == purpose;
    }) : null;

    // 2. Keep the value in a variable
    var currentSelectValue = currentSelectBody ? currentSelectBody.value : null;

    // 3. Triggers callbacks on user action
    var addTag = function(evt) {
        if (currentSelectBody) {
            args.onUpdateBody(currentSelectBody, {
            type: 'TextualBody',
            purpose: purpose,
            value: evt.target.value
           });
        } else { 
          args.onAppendBody({
            type: 'TextualBody',
            purpose: purpose,
            value: evt.target.value
          });
        }
    }

    if (!currentSelectValue) {
        // console.log("currentSelectValue is null --> let select first option");
        const options = _.get(args, 'options', []);
        if (options && options.length > 0) {
            currentSelectValue = options[0]; //select first option
            args.onAppendBody({
                type: 'TextualBody',
                purpose: purpose,
                value: currentSelectValue
          });
        }
    }

    var createOptions = function(options){
        var html$ = '';
        _.forEach(options, (option)=>{
            if(_.isEqual(option,currentSelectValue)){
                html$ += '<option value="'+option+'" selected>'+option+'</option>';
            }
            else{
                html$ += '<option value="'+option+'">'+option+'</option>';
            }
        });
        // console.log("createOptions -> " + html$);
        return html$;
    }

    // 4. This part renders the UI elements
    var createSelect = function(){
        var select = document.createElement('select');
        select.className = 'form-control';
        select.addEventListener('change', addTag); 
        let html$ = '';
        const options = _.get(args, 'options', []);
        const parent = _.get(args, 'parent', false)
        if(!parent){
            html$ = createOptions(options);
        }else{
            const parentBody =  args.annotation.bodies.find((b) => {
                return b.purpose == parent;
            });
            if(parentBody && parentBody.value){
                const choices = _.get(args, 'options.'+parentBody.value, []);
                html$ = createOptions(choices);
            }else{
                html$ = createOptions([]);
            }
        }
        select.innerHTML = html$;
        // console.log("createSelect -> " + html$);
        return select;
    }

    var container = document.createElement('div');
    container.className = 'select-widget';
    container.appendChild(createSelect());
    return container;
}