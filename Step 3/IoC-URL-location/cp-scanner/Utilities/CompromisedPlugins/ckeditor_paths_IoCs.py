import re

# https://www.biomedia.asia/assets/14a67d8d/ckeditor/ckeditor.js

# 'Copyright (c) 2003-2013, CKSource - Frederico Knabben. All rights reserved.
COPYRIGHT_RE = re.compile(r'Copyright \([C|c]\) (\d{4}\s*-\s*\d{4}).*?Frederico (?:Caldeira )?Knabben.(?: All rights reserved)?', re.S)
# 'timestamp:"F0RD",version:"4.4.7",revision:"3a35b3d"'
JS_VERSION_1_RE = re.compile(r'version\s*:\s*["|\']{1}(\d(\.\d+)*)["|\']{1}\s*,\s*revision\s*:\s*["|\']{1}(\w+)*["|\']{1}', re.S)
# ## CKEditor 4.0
CHANGES_MD_RE = re.compile(r'(###)?\s*CKEditor\s*(\d(\.\d+)*)', re.S)
# "CKFinder.version='2.3';CKFinder.revision='2464'"
README_CHANGELOG_1_RE = re.compile(r'CKFinder\.version\s*=\s*["|\']{1}(\d(\.\d+)*)["|\']{1}\s*;\s*CKFinder\.revision\s*=\s*["|\']{1}(\d+)*["|\']{1}', re.S)
# <title>CKFinder 1.4</title>
TITLE_RE = re.compile(r'<title>\s*(CKFinder \d(\.\d*)*)\s*</title>', re.S)
# ### Version 2.3.1
README_CHANGELOG_2_RE = re.compile(r'###\s*Version\s*(\d(\.\d+)*)', re.S) # take the first one
# <h2 id="version-3-4-4">Version 3.4.4</h2>
README_CHANGELOG_3_RE = re.compile(r'<h2 id=["|\']{1}version(.*?)["|\']{1}>Version\s*(\d(\.\d+)*)</h2>', re.S) #take the first match!
# @version 2.51
JS_VERSION_3A_RE = re.compile(r'@version\s*(\d(\.\d+)*)', re.S)
# @package KCFinder
JS_VERSION_3B_RE = re.compile(r'@package\s*[a-zA-Z]+', re.S)
# @copyright 2010, 2011 KCFinder Project
JS_VERSION_3C_RE = re.compile(r'@copyright\s*(\d{4}\s*,\s*\d{4})', re.S)
# "window.FCKeditorAPI = {Version : "2.6.3",VersionBuild : "19836"
JS_VERSION_4_RE = re.compile(r'(window\.)?FCKeditorAPI\s*=\s*\{\s*Version\s*:\s*["|\']{1}(\d(\.\d+)*)["|\']{1},\s*VersionBuild\s*:\s*["|\']{1}(\d+)["|\']{1}', re.S)
# FCKeditorAPI={\ Version:\'2.3.2\',\ VersionBuild:\'1082\'
JS_VERSION_5_RE = re.compile(r'FCKeditorAPI\.Version=.*?(\d(\.\d+)*)', re.S)
# FCKeditorAPI.Version='2.1.1
JS_VERSION_6_RE = re.compile(r'FCKeditorAPI\.Versions*=\s*["|\']{1}(\d(\.\d+)*)["|\']{1}', re.S)


paths_iocs = {

    'CKEditor': {
        'ckeditor.js': {
            # "flags": ['timestamp:"F0RD",version:"4.4.7",revision:"3a35b3d"',
            #              'Copyright (c) 2003-2013, CKSource - Frederico Knabben. All rights reserved.'],
            "flags": [JS_VERSION_1_RE, COPYRIGHT_RE],
            'entry_points': ['', 'js', 'public', 'admin', 'sites/all/modules/'],
            'endpoints': ['ckeditor.js', 'ckeditor/ckeditor.js'],
        },
        # 'ckeditor.html': {
        #     'flags': '',
        #     'paths': ['/base/js/ckfinder/_samples/js/ckeditor.html']
        # },
        'README.md': {
            # "flags": ['CKEditor 4'],
            "flags": [CHANGES_MD_RE],
            'entry_points': ['js'],
            'endpoints': ['ckeditor/README.md']
        },
        'CHANGES.md': {
            # "flags": ['CKEditor 4'],
            "flags": [CHANGES_MD_RE],
            'entry_points': ['js', ''],
            'endpoints': ['ckeditor/CHANGES.md']
        }
    },

    'CKfinder': {
        'ckfinder.js': {
            # 'flags': ["CKFinder.version='2.3';CKFinder.revision='2464'",
            #           'Copyright (c) 2003-2018, CKSource - Frederico Knabben. All rights reserved.'],
            'flags': [README_CHANGELOG_1_RE, COPYRIGHT_RE],
            'entry_points': ['', 'admin', 'ckeditor', 'js'],
            'endpoints': ['ckfinder/ckfinder.js'],
        #},
        # 'ckfinder.html': {
        #     # 'flags': ["<title>CKFinder 1.4</title>"],
        #     'flags': [TITLE_RE],
        #     'entry_points': ['', 'assets-admin/js', 'assets/admin/js', 'assets/developers/js', 'lazyweb/libs', 'js',
        #                      'static/admin/js', 'public'],
        #     'endpoints': ['ckeditor/ckfinder.html', 'ckfinder_3.4.1/ckfinder.html', 'ckdrive/ckfinder.html']
        },'changelog.txt': {
            # 'flags': ["Copyright (C) 2007-2013, CKSource - Frederico Knabben. All rights reserved.",
            #              '### Version 2.3.1', '<h2 id="version-3-4-4">Version 3.4.4</h2>'],
            'flags': [COPYRIGHT_RE, README_CHANGELOG_2_RE, README_CHANGELOG_3_RE],
            'entry_points': ['', 'js', 'public'],
            'endpoints': ['ckfinder/changelog.txt', 'ckfinder/CHANGELOG.html']
        }
    },

    'KCFinder': {
        # 'browse.php': { # init is more important!
        #     'flags': "",
        #     'entry_points': ['app/webroot/js', '', 'admin/ckeditor', 'themes/js', 'blog/ckeditor_fullcolor'],
        #     'endpoints': ['kcfinder_254/browse.php', 'kcfinder_251/browse.php', 'kcfinder/browse.php']
        # },
        'init.js':{
            # 'flags': ['@version 2.51', '@package KCFinder', '@copyright 2010, 2011 KCFinder Project'],
            'flags': [JS_VERSION_3A_RE, JS_VERSION_3B_RE, JS_VERSION_3C_RE],
            'entry_points': ['app/webroot/js', 'admin/ckeditor', ''],
            'endpoints': ['kcfinder_251/js/browser/init.js', 'kcfinder/js/browser/init.js']
        }
    },
    'FCKEditor':{
        'test.html': {
            # 'flags': ['Copyright (C) 2003-2007 Frederico Caldeira Knabben'],
            'flags': [COPYRIGHT_RE],
            'entry_points': ['admin', '/phplist/admin', '', 'sites/all/modules/fckeditor'],
            'endpoints': ['fckeditor/editor/filemanager/browser/default/connectors/test.html', 'fckeditor/editor/filemanager/connectors/test.html',
                          'FCKeditor/editor/filemanager/browser/default/connectors/test.html', 'FCKeditor/editor/filemanager/connectors/test.html']
                        # 'fckeditor/editor/filemanager/connectors/uploadtest.html', 'FCKeditor/editor/filemanager/connectors/uploadtest.html'

        },
        'config.py': { # https://www.elalmacen.cl/avanza/ADMIN/fckeditor/editor/filemanager/connectors/py/upload.py htaccess.txt
            # 'flags': ['Copyright (C) 2003-2008 Frederico Caldeira Knabben'],
            'flags': [COPYRIGHT_RE],
            'entry_points': ['avanza/ADMIN', 'admin', ''],
            'endpoints': ['fckeditor/editor/filemanager/connectors/py/config.py']
         },
        # 'upload.php': {
        #     'flags': [],
        #     'entry_points': ['admin', ''],
        #     'endpoints': ['fckeditor/editor/filemanager/connectors/php/upload.php']
        # },
        'fckeditorcode_gecko.js': { # seems to be less important?
            # 'flags': ['"window.FCKeditorAPI = {Version : "2.6.3",VersionBuild : "19836"', 'Copyright (C) 2003-2007 Frederico Caldeira Knabben'],
            'flags': [JS_VERSION_4_RE,COPYRIGHT_RE],
            'entry_points': ['', 'admin', 'sites/all/modules/fckeditor'],
            'endpoints': ['FCKeditor/editor/js/fckeditorcode_gecko.js', 'ckeditor/editor/js/fckeditorcode_gecko.js',
                          'fckeditor/editor/js/fckeditorcode_gecko.js']
        },
        'fck_startup.js': {
            # 'flags': ["FCKeditorAPI.Version='2.1.1'", 'Copyright (C) 2003-2007 Frederico Caldeira Knabben'],
            'flags': [JS_VERSION_6_RE, COPYRIGHT_RE],
            'entry_points': ['', 'admin'],
            'endpoints': ['FCKeditor/editor/js/fck_startup.js']
        }
    }
}
