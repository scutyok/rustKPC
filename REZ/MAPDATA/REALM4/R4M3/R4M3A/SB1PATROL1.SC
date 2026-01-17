Name "sb1patrol1"

AbortOnDamage			FALSE		//	abort the script if we're damaged
AbortOnEnemySighted		FALSE		//	Don't abort if we sight an enemy
CanBeMoved			FALSE
SightRange				1400.0		//	can see 1024 units
TimeOut					999.0
WalkSpeed				70
YawRate					180
CheckForEnemyInterval	0.5			//	check for enemies every 1/2 second
TargetPlayer						//	target the player

Loop1:
OnEnemyOutOfRange_Goto	DISABLE		//	don't branch on enemy out of range
OnTimerExpired_Goto		DISABLE	

OnEnemySighted_Goto		Attack				//	if we see the player, jump to Attack
OnDamage_Goto			FarAttack			//	if we get hit, jump to FarAttack
TimeOut		10.0

//	walk to the path3 and wait
Walk
StartAnim	"Walk"
TurnTo		path3
MoveTo		path3
TurnToAngle	180

//	if the player is in range, attack
StartAnim	"AmbA"
Wait		2.0

//	turn to look at path2 ...
StartAnim	"Walk"
TurnTo		path2

//	then walk to path 2
StartAnim	"Walk"
MoveTo		path2
TurnToAngle	180

//	wait here for a bit
StartAnim	"AmbA"
Wait		2.0

//	walk to path1
Walk
YawRate		180
StartAnim	"Walk"
TurnTo		path1
MoveTo		path1

//	turn to face off of ledge
TurnToAngle	180
StartAnim	"AmbA"
Wait		2.0

//	walk back to path2 and start all over again
StartAnim	"Walk"
TurnTo		path2
MoveTo		path2

//	turn an look off of the ledge again
TurnToAngle	180
StartAnim	"AmbA"
Wait		2.0

Goto	Loop1

FarAttack:
SightRange	2048.0

Attack:
OnDamage_Goto			DISABLE	//	don't branch if damaged while attacking
OnEnemySighted_Goto		DISABLE	//	don't branch if enemy sighted while attacking
OnEnemyOutOfRange_Goto	Loop1

YawRate		540
TimeOut		2.5

//	choose a random place to start from
if_RandomLessThan25_Goto	Attack_path2
if_RandomLessThan25_Goto	Attack_path1
if_RandomLessThan25_Goto	Attack_path3

//
//	Attack 2
//
Attack_path2:
TurnTo		path2
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path2a
StartRandomTimer	0.5		2.0

MoveTo		path2

Attack_path2a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path3

//
//	Attack 1
//
Attack_path1:
TurnTo		path1
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path1a
StartRandomTimer	0.5		2.0

MoveTo		path1

Attack_path1a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path2

//
//	Attack 3
//
Attack_path3:
TurnTo		path3
Run
StartAnim	"Run"

//	start a random timer that will make us stop sometime between when we start
//	running and when we reach the path...
OnTimerExpired_Goto		Attack_path3a
StartRandomTimer	0.5		2.0

MoveTo		path3

Attack_path3a:
TurnToTarget	0.33	//	this is the time start of attack to firing in anim
RangedAttackTarget

if_RandomLessThan50_Goto	Attack_path1

Goto	Attack_path2

