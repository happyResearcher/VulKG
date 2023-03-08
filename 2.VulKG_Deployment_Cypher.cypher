//Project: VulKG Project
//DBMA: Graph DBMS
//DATABASE: neo4j
//Password: Neo4j

// ###############  import/VulnerabilityNodes.csv  ####################
// create entity
// label: Vulnerability

// set uniqueness constraint
CREATE CONSTRAINT UniqueCveID ON (v:Vulnerability) ASSERT v.cveID IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create Entity with properties: Vulnerability (no relationships)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///VulnerabilityNodes.csv' )
 YIELD map AS row RETURN row",
 "WITH 
	 row.cveID as cveID, 
	 date(row.publishedDate) AS publishedDate,
	 row.description_value AS description,
	 toInteger(row.num_reference) AS numOfReference,
	 toInteger(row.v2version) AS v2version,
	 toFloat(row.v2baseScore) AS v2baseScore,
	 row.v2accessVector AS v2accessVector,
	 row.v2accessComplexity AS v2accessComplexity,
	 row.v2authentication AS v2authentication,
	 row.v2confidentialityImpact AS v2confidentialityImpact,
	 row.v2integrityImpact AS v2integrityImpact,
	 row.v2availabilityImpact AS v2availabilityImpact,
	 row.v2vectorString AS v2vectorString,
	 toInteger(row.v2impactScore) AS v2impactScore,
	 toInteger(row.v2exploitabilityScore) AS v2exploitabilityScore,
	 toBoolean(row.v2userInteractionRequired) AS v2userInteractionRequired,
	 row.v2severity AS v2severity,
	 toBoolean(row.v2obtainUserPrivilege) AS v2obtainUserPrivilege,
	 toBoolean(row.v2obtainAllPrivilege) AS v2obtainAllPrivilege,
	 toBoolean(row.v2acInsufInfo) AS v2acInsufInfo,
	 toBoolean(row.v2obtainOtherPrivilege) AS v2obtainOtherPrivilege,
	 toFloat(row.v3version) AS v3version,
	 toInteger(row.v3baseScore) AS v3baseScore,
	 row.v3attackVector AS v3attackVector,
	 row.v3attackComplexity AS v3attackComplexity,
	 row.v3privilegesRequired AS v3privilegesRequired,
	 row.v3userInteraction AS v3userInteraction,
	 row.v3scope AS v3scope,
	 row.v3confidentialityImpact AS v3confidentialityImpact,
	 row.v3integrityImpact AS v3integrityImpact,
	 row.v3availabilityImpact AS v3availabilityImpact,
	 row.v3vectorString AS v3vectorString,
	 toInteger(row.v3impactScore) AS v3impactScore,
	 toInteger(row.v3exploitabilityScore) AS v3exploitabilityScore,
	 row.v3baseSeverity AS v3baseSeverity
 MERGE (v:Vulnerability {cveID:cveID})
    ON CREATE SET 
	v.publishedDate=publishedDate,
	v.description =description,
	v.numOfReference =numOfReference,
	v.v2version=v2version,
	v.v2baseScore=v2baseScore,
	v.v2accessVector=v2accessVector,
	v.v2accessComplexity=v2accessComplexity,
	v.v2authentication=v2authentication,
	v.v2confidentialityImpact=v2confidentialityImpact,
	v.v2integrityImpact=v2integrityImpact,
	v.v2availabilityImpact=v2availabilityImpact,
	v.v2vectorString=v2vectorString,
	v.v2impactScore=v2impactScore,
	v.v2exploitabilityScore=v2exploitabilityScore,
	v.v2userInteractionRequired=v2userInteractionRequired,
	v.v2severity=v2severity,
	v.v2obtainUserPrivilege=v2obtainUserPrivilege,
	v.v2obtainAllPrivilege=v2obtainAllPrivilege,
	v.v2acInsufInfo=v2acInsufInfo,
	v.v2obtainOtherPrivilege=v2obtainOtherPrivilege,
	v.v3version=v3version,
	v.v3baseScore=v3baseScore,
	v.v3attackVector=v3attackVector,
	v.v3attackComplexity=v3attackComplexity,
	v.v3privilegesRequired=v3privilegesRequired,
	v.v3userInteraction=v3userInteraction,
	v.v3scope=v3scope,
	v.v3confidentialityImpact=v3confidentialityImpact,
	v.v3integrityImpact=v3integrityImpact,
	v.v3availabilityImpact=v3availabilityImpact,
	v.v3vectorString=v3vectorString,
	v.v3impactScore=v3impactScore,
	v.v3exploitabilityScore=v3exploitabilityScore,
	v.v3baseSeverity=v3baseSeverity
 RETURN count(*)",
{batchSize: 500}
);



// ###############  import/ExploitNodes.csv  ####################
// create entity with properties:: Exploit
// label:Exploit

// create uniqueness constraint
CREATE CONSTRAINT UniqueEID ON (e:Exploit) ASSERT e.eid IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

// create entity: Vulnerability (no relationships)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///ExploitNodes.csv')
 YIELD map AS row RETURN row",
 "WITH 
 row.ExploitID AS eid,
 date(row.Exploit_Date) AS exploitPublishDate,
 row.Author	AS author,
 row.Exploit_Type AS exploitType,
 row.Platform AS platform
 MERGE (e:Exploit {eid:eid})
    ON CREATE SET 
	e.exploitPublishDate=exploitPublishDate,
	e.exploitType = exploitType,
	e.platform = platform
 RETURN count(*)",
 {batchSize: 500}
);


//###############  import/ExploitNodes.csv  ####################
// create entity
// label: Author

// set uniqueness constraint
CREATE CONSTRAINT UniqueAuthorName ON (a:Author) ASSERT a.authorName IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

// create entity with properties: Author (no relationships)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///ExploitNodes.csv')
 YIELD map AS row RETURN row",
 "WITH
 row.Author AS authorName
 MERGE (a:Author {authorName:authorName})
 ON CREATE SET
	a.authorName=authorName
 RETURN count(*)",
 {batchSize: 500}
);


// ###############  import/ExploitNodes.csv   ####################
// create relationship:
// relationship type: WRITES (Author WRITES Vulnerability)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///ExploitNodes.csv')
 YIELD map AS row RETURN row",
 "WITH
	 row.ExploitID AS eid,
	 row.Author	AS authorName
 MATCH (e:Exploit {eid:eid})
 MATCH (a:Author {authorName:authorName})
 MERGE (a)-[r:WRITES]->(e)
 RETURN count(r)",
 {batchSize: 500}
);


// ###############  import/Vulnerability_HAS_EXPLOIT_Exploit_relationship.csv   ####################
// create relationship:
// relationship type: EXPLOITS (Exploit EXPLOITS Vulnerability)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///Vulnerability_HAS_EXPLOIT_Exploit_relationship.csv')
 YIELD map AS row RETURN row",
 "WITH 
	 row.eid AS eid,
	 row.cveID	AS cveID
 MATCH (e:Exploit {eid:eid})
 MATCH (v:Vulnerability {cveID:cveID})
 MERGE (e)-[r:EXPLOITS]->(v)
 RETURN count(r)",
 {batchSize: 500}
);

// ###################### import/WeaknessNodes.csv   ####################
// create entity
// entity label: Weakness

// uniqueness constraint  ---DONE
CREATE CONSTRAINT UniquecweID ON (w:Weakness) ASSERT w.cweID IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create entity with properties: Weakness (no relationships)
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///WeaknessNodes.csv')
 YIELD map AS row RETURN row",
 "WITH
 row.cweID AS cweID,
 split(row.cweView,',') AS cweView,
 row.cweName	AS cweName,
 row.weaknessAbstraction AS weaknessAbstraction,
 row.status AS status,
 row.description AS description,
 row.extendedDescription AS extendedDescription
 MERGE (w:Weakness {cweID:cweID})
    ON CREATE SET
	w.cweView=cweView,
	w.cweName = cweName,
	w.weaknessAbstraction = weaknessAbstraction,
	w.status = status,
	w.description = description,
	w.extendedDescription = extendedDescription
 RETURN count(*)",
 {batchSize: 500}
);


// ###################### import/VulnerabilityNodesAddProperties.csv  ##########################
// create relationship
// relationship type: EXAMPLE_OF (Vulnerability EXAMPLE_OF Weakness)

CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///VulnerabilityNodesAddProperties.csv')
 YIELD map AS row RETURN row",
 "WITH
	 row.cveID AS cveID,
	 row.CWEID AS cweID
 MATCH (v:Vulnerability {cveID:cveID})
 MATCH (w:Weakness {cweID:cweID})
 MERGE (v)-[r:EXAMPLE_OF]->(w) // add relationships
 RETURN * ",
 {batchSize: 500}
);


// #######################  import/VulnerabilityNodesAddProperties.csv        ####################
// add properties to entity
// entity label: Vulnerability
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///VulnerabilityNodesAddProperties.csv')
 YIELD map AS row RETURN row",
 "WITH 
	 row.cveID AS cveID,
	 row.GainedAccess AS gainedAccess,
	 split(row.VulnerabilityType,',') as vulnerabilityType
 MERGE (v:Vulnerability {cveID:cveID})
    ON MATCH SET
	v.gainedAccess=gainedAccess,
	v.vulnerabilityType = vulnerabilityType
 RETURN count(*)",
 {batchSize: 500}
);

// ###################### import/DomainNodes_Vulnerability_HAS_REFERENCE_Domain_relationship.csv     ##########################
// create entity and relationship
// entity label: Domain
// relationship type: REFERS_TO (Vulnerability REFERS_TO Domain)

// uniqueness constraint
CREATE CONSTRAINT UniqueDomainName ON (d:Domain) ASSERT d.domainName IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create Domain entity and REFERS_TO relationship
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///DomainNodes_Vulnerability_HAS_REFERENCE_Domain_relationship.csv')
 YIELD map AS row RETURN row",
 "WITH 
	 row.cveID AS cveID,
	 row.domainName AS domainName
 MERGE (d:Domain {domainName:domainName}) //add nodes
 WITH *
 MATCH (v:Vulnerability {cveID:cveID})
 MERGE (v)-[r:REFERS_TO]->(d) // add relationships
 RETURN * ",
 {batchSize: 500}
);

// ################### import/ProductNodes_VendorNodes_Vulnerability_AFFECTS_Product_BELONGS_TO_Vendor.csv #######################################
// create entity and relationship
// entity label: Product
// relationship type: AFFECTS (Vulnerability AFFECTS Product)

// uniqueness constraint
CREATE CONSTRAINT UniqueProductName ON (p:Product) ASSERT p.productName IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

////create Product entity and AFFECTS relationship
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///ProductNodes_VendorNodes_Vulnerability_AFFECTS_Product_BELONGS_TO_Vendor.csv')
 YIELD map AS row RETURN row",
 "WITH 
	 row.cveID AS cveID,
	 row.Product AS productName, 
	 row.ProductType AS productType,
	 toInteger(row.Nversions) AS numOfVersion
 MERGE (p:Product {productName:productName}) //add nodes
 ON CREATE SET p.productType=productType
 WITH *
 MATCH (v:Vulnerability {cveID:cveID})
 MERGE (v)-[r:AFFECTS]->(p) // add relationships
 ON CREATE SET r.numOfVersion=numOfVersion
 RETURN * ",
 {batchSize: 500}
);

// ##################### import/ProductNodes_VendorNodes_Vulnerability_AFFECTS_Product_BELONGS_TO_Vendor.csv #######################################
// create entity and relationship
// entity label: Vendor
// relationship type: BELONGS_TO (Product BELONGS_TO Vendor)


// uniqueness constraint  
CREATE CONSTRAINT UniqueVendorName ON (v:Vendor) ASSERT v.vendorName IS UNIQUE;

// verify the creation of constraint
CALL db.constraints;

//create Vendor AND BELONGS_TO  relationship
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///ProductNodes_VendorNodes_Vulnerability_AFFECTS_Product_BELONGS_TO_Vendor.csv')
 YIELD map AS row RETURN row",
 "WITH 
	 row.Product AS productName,
	 row.Vendor AS vendorName
 MERGE (vd:Vendor {vendorName:vendorName}) //add nodes
 WITH *
 MATCH (p:Product {productName:productName})
 MERGE (p)-[r:BELONGS_TO]->(vd) // add relationships
 RETURN * ",
 {batchSize: 500}
);

//##################### import/add affectedVersion property to AFFECTS relationship ####################################
// add affectedVersion property to AFFECTS relationships
// add new null list affectedVersion property for all AFFECTS relationships
MATCH ()-[r:AFFECTS]->()
SET r.affectedVersion=[]
RETURN count(r); //reture: 212656

// add affectedVersion
CALL apoc.periodic.iterate(
"CALL apoc.load.csv('file:///AffectsAddProperty.csv')
 YIELD map AS row RETURN row",
 "WITH
   row.cveID AS cveID,
	 row.Product AS productName,
   row.Version AS version
MATCH (v:Vulnerability{cveID:cveID})-[r:AFFECTS]->(p:Product{productName:productName})
	SET r.affectedVersion = r.affectedVersion + [version]
 RETURN count(*)",
 {batchSize: 500}
); //return: 1126644


//###################### add index for fast seaching ############################
// ########### add index for single property
// Vulnerability
CREATE INDEX VulnerabilityV2version FOR (v:Vulnerability) ON (v.v2version);
CREATE INDEX VulnerabilityV3version FOR (v:Vulnerability) ON (v.v3version);
CREATE INDEX VulnerabilityPublishedDate FOR (v:Vulnerability) ON (v.publishedDate);
CREATE INDEX VulnerabilityDescription FOR (v:Vulnerability) ON (v.description);

// Exploit
CREATE INDEX ExploitExploitPublishDate FOR (e:Exploit) ON (e.exploitPublishDate);

// ############# full-text schema index:string values only 
// Vulnerability
CALL db.index.fulltext.createNodeIndex(
      'VulnerabilityDescriptionFullTextSchema',['Vulnerability'], ['description']);

// check the full-text schema index
CALL db.indexDetails('VulnerabilityDescriptionFullTextSchema');


// ############## delete vulnerabilities with **Reject** in description ###############
// search
MATCH (n:Vulnerability)
WHERE n.description STARTS WITH '** REJECT **'
RETURN n.cveID, n.description;  //8950 records

// delete
MATCH (n:Vulnerability)
WHERE n.description STARTS WITH '** REJECT **'
DETACH DELETE n
RETURN count(*); //Deleted 8950 nodes, deleted 191 relationships

//################ add new properties to Vulnerability ############################
// property name: exploitability, exploitDate

// initialise exploitability and exploitDate
MATCH (v1:Vulnerability)
SET v1.exploitability=0, v1.exploitDate=[]
RETURN COUNT(*); // 148609

// set exploitability and exploitDate based on EXPLOITS relationships
MATCH (v1:Vulnerability)<-[r:EXPLOITS]-(e:Exploit)
SET v1.exploitability=1, v1.exploitDate=v1.exploitDate + [e.exploitPublishDate]
RETURN COUNT(*); // 28791, Set 57582 properties

// ############### explore exploitable vulnerabilities #######################
// return number of exploitable vulnerabilities
MATCH (v1:Vulnerability)
WHERE v1.exploitability=1
RETURN COUNT(*); // 23660

// Return number of vulnerabilities having only 1 exploit
MATCH (v1:Vulnerability)
WHERE size(v1.exploitDate)=1
RETURN COUNT(*); //20689

// Return number of vulnerabilities having more than 1 exploit
MATCH (v1:Vulnerability)
WHERE size(v1.exploitDate)>1
RETURN COUNT(*); //2971

// ############## generate fig 1
CALL db.schema.visualization()

// ############### get results in TABLE IV: Statistics of vulnerability knowledge graph ##############
MATCH (n:Vulnerability) RETURN count(n); //148609
MATCH (n:Exploit) RETURN count(n);  //43743
MATCH (n:Weakness) RETURN count(n); //926
MATCH (n:Product) RETURN count(n); //44269
MATCH (n:Vendor) RETURN count(n); //20347
MATCH (n:Author) RETURN count(n); //9204
MATCH (n:Domain) RETURN count(n); //9578

MATCH p=()-[r:EXPLOITS]->() RETURN count(p); //28791
MATCH p=()-[r:AFFECTS]->() RETURN count(p); //212654
MATCH p=()-[r:BELONGS_TO]->() RETURN count(p); //47509
MATCH p=()-[r:EXAMPLE_OF]->() RETURN count(p); //71031
MATCH p=()-[r:WRITES]->() RETURN count(p); //43743
MATCH p=()-[r:REFERS_TO]->() RETURN count(p); //429728
