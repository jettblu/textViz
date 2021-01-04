from django.shortcuts import render, redirect
from django.core import serializers
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import login, logout, authenticate
from django.contrib import messages
from django.http import JsonResponse
from .textAnalysis import readTextsWeb as readTextsWeb
from .textAnalysis import utils as utils
from .textAnalysis import visuals as visualize
from .textAnalysis import messageAnalysis as messageAnalysis
from .models import MessageDocument
from .forms import NewUserForm
from django.core.paginator import Paginator
import json


def homepage(request):
    # Handle file upload
    if request.method == 'POST':
        # ensures user is logged in
        if request.user.is_authenticated:
            # checks if file is .xml
            if request.FILES['msgFile'].name.lower().endswith('.xml'):
                messages.success(request, "File processing...")
                newdoc = MessageDocument(msgFile=request.FILES['msgFile'], documentOwner=request.user)
                newdoc.save()
                contacts = readTextsWeb.readTextsfromXml(newdoc.msgFile.path)
                newdoc.contacts=contacts
                newdoc.save()
                messages.success(request, "Nice! File processed.")
                return redirect('main:display', userName=request.user.username, fileKey=newdoc.pk)
            else:
                messages.error(request, "Please submit a valid xml file.")
        else:
            messages.error(request, "You must be logged in to analyze your text messages!")

    # Load documents for the list page
    documents = MessageDocument.objects.all()

    context = {'documents': documents}

    # Render list page with the documents and the form
    return render(request=request,
                  template_name="main/home.html",
                  context=context)


def display(request, userName, fileKey):
    numContacts = 10
    file = MessageDocument.objects.get(pk=fileKey)
    contacts = file.contacts
    sortedContacts = utils.sortContactFrequency(contacts)
    paginator = Paginator(sortedContacts, numContacts)  # Show n contacts per page.
    page_number = request.GET.get('page')
    # controls how many contacts are displayed in text count graph
    if page_number is not None:
        toGraph = numContacts*int(page_number)
    else:
        toGraph = 10
    plot_div = visualize.allTextCounts(contactsDict=contacts, nContacts=toGraph)

    page_obj = paginator.get_page(page_number)
    contactNameList = page_obj.object_list
    contactObj = contacts[contactNameList[0]]

    # check to see if these contacts have already been processed
    if contactObj.incoming.avgLag is None and contactObj.outgoing.avgLag is None:
        for contactName in contactNameList:
            contact = contacts[contactName]
            readTextsWeb.organizeContact(contact)
            readTextsWeb.calculateLag(contact)
        file.contacts = contacts
        file.save()
    return render(request=request,
                  template_name="main/display.html",
                  context={'contacts': contacts, 'sorted': sortedContacts, 'fileKey':
                           fileKey, 'plot_div': plot_div, 'page_obj': page_obj})


def analysis(request, userName, fileKey, contactName):
    file = MessageDocument.objects.get(pk=fileKey)
    contacts = file.contacts
    contact = contacts[contactName]
    # only calculate sentiment score if hasn't been calculated yet
    if contact.sentimentScore is None and contact.incoming.sentimentScore is None:
        readTextsWeb.calculateSentiment(contact)
        file.contacts = contacts
        file.save()
    print(contact)
    return render(request=request,
                  template_name="main/analysis.html",
                  context={'contact': contact, 'fileKey': fileKey, 'contactName': contactName})


# <!--Login view code taken from https://pythonprogramming.net/user-login-logout-django-tutorial/-->
def loginRequest(request):
    if request.method == 'POST':
        form = AuthenticationForm(request=request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.info(request, f'You are now logged in as {username}')
                return redirect("main:homepage")
            else:
                messages.error(request, "Invalid username or password")
        else:
            messages.error(request, "Invalid username or password")

    form = AuthenticationForm()
    return render(request, "main/login.html", {"form": form})


def account(request, userName):
    user = request.user
    files = user.files.all()
    incomingTextCount, outgoingTextCount, allTexts = 0, 0, 0
    # get summary stats for all files associated with user
    for file in files:
        contacts = file.contacts
        incomingTextCount, outgoingTextCount, allTexts = utils.allContactsSummmary(contacts)
    return render(request=request, template_name='main/account.html', context={'files': files,
                                                                               'incomingTextCount':incomingTextCount,
                                                                               'outgoingTextCount':outgoingTextCount,
                                                                               'allTexts': allTexts})


def experimental(request):
    displayResults = False
    if request.method == 'POST':
        displayResults = True
        text = request.POST.get('text')
        sentiment, confidence, tokens = messageAnalysis.experimentalIsPositive(text)
        return render(request=request, template_name='main/experimental.html', context={'display': displayResults,
                                                                                        'sentiment': sentiment,
                                                                                        'confidence': confidence,
                                                                                        'tokens': tokens})
    return render(request=request, template_name='main/experimental.html', context={'display': displayResults})


# '''Login code taken from https://pythonprogramming.net/user-login-logout-django-tutorial/'''
def register(request):
    if request.method == 'POST':
        form = NewUserForm(request.POST)
        if form.is_valid():
            user = form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'New Account Created: {username}')
            login(request, user)
            messages.info(request, f'You are now logged in as {username}')
            return redirect("main:homepage")
        else:
            for msg in form.error_messages:
                messages.error(request, f"{msg}: form.error_messages[msg]")

    form = NewUserForm
    return render(request, "main/register.html", context={"form": form})


# '''Logout code taken from https://pythonprogramming.net/user-login-logout-django-tutorial/'''
def logoutRequest(request):
    logout(request)
    messages.info(request, 'Logged out successfully!')
    return redirect("main:homepage")


# uncomment when implementing ajax load
'''def updateContactDropdown(request):
    print("HERE")
    fileKey = request.GET.get('fileKey', None)
    file = MessageDocument.objects.get(pk=fileKey)
    contacts = file.contacts

    contactName = request.GET.get('contactName', None)
    contact = contacts[contactName]

    # destructively modifies contact object
    readTextsWeb.organizeContact(contactObj=contact)
    readTextsWeb.calculateLag(contact=contact)
    data = {
        'contacts': contacts
    }
    return JsonResponse(data)'''


def visuals(request, userName, fileKey, contactName):
    file = MessageDocument.objects.get(pk=fileKey)
    contacts = file.contacts
    sortedContacts = utils.sortContactFrequency(contacts)
    contact = contacts[contactName]
    plots = visualize.visualizeContact(contact)
    return render(request=request,
                  template_name="main/visuals.html",
                  context={'contacts': contacts, 'sorted': sortedContacts, 'fileKey': fileKey,
                            'contactName': contactName, 'plots': plots})


def info(request):
    return render(request=request, template_name="main/info.html")
